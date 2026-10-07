#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "infernix/ops/target_logprobs.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      prompt_attention(plan.prompt_attention), draft_window(plan.draft_window),
      neural_draft_window(plan.neural_draft_window), tree_widths(plan.tree_widths),
      draft_tree_paths(plan.draft_tree_paths), ngram_draft_window(plan.ngram_draft_window),
      ngram_min_match(plan.ngram_min_match), speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      round_host(plan.causal_scoring ? std::nullopt
                                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
      // Target log-probabilities and three [kMaximumScoreTopTokens x kCausalScoreTile] distribution
      // regions (causal_score).
      score_logprobs_host(plan.causal_scoring
                              ? std::make_optional<PinnedHostBuffer>(
                                    (1 + 3 * std::size_t{kMaximumScoreTopTokens}) *
                                    kCausalScoreTile * sizeof(float))
                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)},
      prefill_gpu_timer_(device_in) {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    if (tree_widths.automatic_mode()) {
        std::array<std::vector<std::uint32_t>, kMaximumConcurrency> trees;
        for (std::uint32_t b = 1; b <= max_concurrency; ++b) {
            for (const std::uint32_t columns : tree_widths.automatic) {
                if (tree_widths.tree(b, columns)) { trees[b - 1U].push_back(columns); }
            }
        }
        tree_controller.emplace(neural_draft_window + 1U, trees);
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder      = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    state_images = std::make_unique<StateImageDevicePool>(backing, plan.persistent.state_images);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
        narrow_replay_views.reserve(plan.round_shapes.size());
        for (const SpeculativeRoundShape& shape : plan.round_shapes) {
            const auto width = static_cast<std::int32_t>(shape.verify_drafts + 1U);
            if (shape.verify_drafts == draft_window ||
                std::any_of(narrow_replay_views.begin(), narrow_replay_views.end(),
                            [width](const NarrowReplayView& view) {
                                return view.records.spec.width == width;
                            })) {
                continue;
            }
            GdnReplayRecords narrowed = replay_records->narrowed(width);
            ops::GdnReplayFoldPlan fold(narrowed, state_images->linear().all_layers_view());
            narrow_replay_views.push_back(NarrowReplayView{narrowed, std::move(fold)});
        }
    }
    round_families.reserve(plan.round_shapes.size());
    for (const SpeculativeRoundShape& shape : plan.round_shapes) {
        round_families.push_back(SpeculativeRoundFamily{shape, {}});
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        auto* local = state_images->dflash_local();
        if (!local) { throw std::logic_error("DFlash StateImage has no local state"); }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    // The hybrid prefix cache owns the whole Host budget as its slab pool (hybrid_program.cpp).
    const auto host_bytes =
        plan.context_cache.enabled && plan.context_cache.mode == ContextCacheMode::Hybrid
            ? std::size_t{0}
            : plan.context_cache.host_capacity_bytes.value();
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry())};
    text_host_kv_page_stride = layouts.front().page_stride;
    if (const auto* backend = backend_kv_cache()) {
        auto layout                 = plan_host_kv_page_layout(backend->page_pool().geometry());
        backend_host_kv_page_stride = layout.page_stride;
        if (layout != layouts.front()) { layouts.push_back(std::move(layout)); }
    }
    std::size_t minimum_stride    = state_images->host_layout().image_bytes;
    std::size_t minimum_kv_stride = layouts.front().page_stride;
    for (const auto& layout : layouts) {
        minimum_stride    = std::min(minimum_stride, layout.page_stride);
        minimum_kv_stride = std::min(minimum_kv_stride, layout.page_stride);
    }
    const auto checked_count = [](std::uint64_t value) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("context resource descriptor capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto logical_states = checked_count(state_images->slot_count() +
                                              host_bytes / state_images->host_layout().image_bytes);
    // One immutable image can serve private replay, an explicit anchor, a public view and an
    // exact-hit endpoint. Alias descriptors are bounded separately from physical state slots.
    checkpoints.resize(checked_count(4ULL * logical_states + 2ULL * max_concurrency));
    const auto address_capacity = checked_count(checkpoints.size() + max_concurrency + 2U);
    if (host_bytes) {
        StartupPhaseScope phase(startup_observer, StartupPhase::HostContextPin,
                                StartupProgressUnit::Bytes, host_bytes);
        host_context_arena = std::make_unique<HostContextArena>(host_bytes, minimum_stride);
        host_state_images =
            std::make_unique<HostStatePool>(*host_context_arena, state_images->host_layout());
        host_kv_arena      = std::make_unique<HostKVArena>(*host_context_arena, layouts);
        const auto extents = checked_count(host_bytes / minimum_kv_stride);
        if (extents) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(*host_kv_arena, extents);
        }
        phase.complete(host_bytes, host_bytes);
    }
    state_store =
        std::make_unique<StateImageStore>(*state_images, host_state_images.get(), logical_states);
    const auto logical_pages = [&](const DeviceKVPagePool& pool) {
        const auto stride = plan_host_kv_page_layout(pool.geometry()).page_stride;
        return checked_count(pool.capacity_pages() + host_bytes / stride);
    };
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_pages(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_pages(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    if (io.constraint.has_value() == causal_scoring) {
        throw std::logic_error("constraint controls do not match the program purpose");
    }
    if (io.constraint) {
        // Every column starts unconstrained (-1); the choice tables are written per request.
        CUDA_CHECK(cudaMemsetAsync(io.constraint->descriptors.data, 0xff,
                                   io.constraint->descriptors.bytes(), device.stream));
        constraint_descriptors_host_.assign(kTokenConstraintColumns, -1);
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    device.synchronize();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        const std::size_t free_before = device.free_bytes();
        prepare_graphs();
        device.synchronize();
        const std::size_t free_after = device.free_bytes();
        graph_measured_bytes         = free_before > free_after ? free_before - free_after : 0;
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
    create_hybrid_prefix_cache(startup_observer);
}

ProgramImpl::~ProgramImpl() noexcept {
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

ScoreResult ProgramImpl::causal_score(PreparedPromptData&& prompt, std::uint32_t first_target,
                                      const ScoreOptions& options) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }
    const std::uint32_t top_k         = options.top_k;
    const std::uint32_t per_position  = options.candidates_per_position;
    const std::int32_t public_tokens  = dimension(parameters.model.resources().public_token_count);
    const std::size_t scored_positions = token_count_size - first_target;
    if (top_k > kMaximumScoreTopTokens || per_position > kMaximumScoreTopTokens ||
        options.candidates.size() != scored_positions * per_position) {
        throw std::invalid_argument("causal score distribution outputs have an invalid shape");
    }
    for (const TokenId candidate : options.candidates) {
        if (candidate < 0 || candidate >= public_tokens) {
            throw std::invalid_argument("causal score candidate is not a public token");
        }
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t required_pages         = kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    ScoreResult output;
    output.logprobs.reserve(scored_positions);
    output.top_ids.reserve(scored_positions * top_k);
    output.top_logprobs.reserve(scored_positions * top_k);
    output.candidate_logprobs.reserve(scored_positions * per_position);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(required_pages, 0, device.stream);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            // The staged columns' first scored position.
            const std::size_t first_position = output.logprobs.size();
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {1, columns});
            Tensor logprobs   = work.alloc(DType::FP32, {1, columns});
            const auto distribution = [&](DType dtype, std::uint32_t rows) {
                return rows == 0 ? Tensor{}
                                 : work.alloc(dtype, {static_cast<std::int32_t>(rows), columns});
            };
            Tensor top_ids            = distribution(DType::I32, top_k);
            Tensor top_logprobs       = distribution(DType::FP32, top_k);
            Tensor candidate_ids      = distribution(DType::I32, per_position);
            Tensor candidate_logprobs = distribution(DType::FP32, per_position);
            Tensor hidden             = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids, public_tokens, logprobs, device.stream);
            // Pinned staging: target log-probabilities, then each distribution output's
            // [kMaximumScoreTopTokens x kCausalScoreTile] region.
            auto* host_logprobs = static_cast<float*>(score_logprobs_host->data());
            auto* host_top_ids  = reinterpret_cast<TokenId*>(host_logprobs + kCausalScoreTile);
            auto* host_top_logprobs = reinterpret_cast<float*>(
                host_top_ids + std::size_t{kMaximumScoreTopTokens} * kCausalScoreTile);
            auto* host_candidate_logprobs =
                host_top_logprobs + std::size_t{kMaximumScoreTopTokens} * kCausalScoreTile;
            CUDA_CHECK(cudaMemcpyAsync(host_logprobs, logprobs.data, logprobs.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream));
            if (top_k > 0) {
                ops::top_logprobs(logits, public_tokens, top_ids, top_logprobs, device.stream);
                CUDA_CHECK(cudaMemcpyAsync(host_top_ids, top_ids.data, top_ids.bytes(),
                                           cudaMemcpyDeviceToHost, device.stream));
                CUDA_CHECK(cudaMemcpyAsync(host_top_logprobs, top_logprobs.data,
                                           top_logprobs.bytes(), cudaMemcpyDeviceToHost,
                                           device.stream));
            }
            if (per_position > 0) {
                CUDA_CHECK(cudaMemcpyAsync(candidate_ids.data,
                                           options.candidates.data() + first_position * per_position,
                                           candidate_ids.bytes(), cudaMemcpyHostToDevice,
                                           device.stream));
                ops::target_logprobs(logits, candidate_ids, public_tokens, candidate_logprobs,
                                     device.stream);
                CUDA_CHECK(cudaMemcpyAsync(host_candidate_logprobs, candidate_logprobs.data,
                                           candidate_logprobs.bytes(), cudaMemcpyDeviceToHost,
                                           device.stream));
            }
            device.synchronize();
            output.logprobs.insert(output.logprobs.end(), host_logprobs,
                                   host_logprobs + staged_columns);
            output.top_ids.insert(output.top_ids.end(), host_top_ids,
                                  host_top_ids + std::size_t{staged_columns} * top_k);
            output.top_logprobs.insert(output.top_logprobs.end(), host_top_logprobs,
                                       host_top_logprobs + std::size_t{staged_columns} * top_k);
            output.candidate_logprobs.insert(
                output.candidate_logprobs.end(), host_candidate_logprobs,
                host_candidate_logprobs + std::size_t{staged_columns} * per_position);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, *state_images, nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, prompt_attention},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.logprobs.size() != scored_positions) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    out.weights = ArenaMemorySummary{weights.device_capacity_bytes, weights.device_capacity_bytes,
                                     weights.device_capacity_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_measured_bytes    = graph_measured_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) { out.host_state_occupied_slots = host_state_images->occupied(); }
    if (host_kv_arena) { out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes(); }
    if (host_context_arena) {
        out.host_context_capacity_bytes = host_context_arena->capacity_bytes();
        out.host_context_occupied_bytes = host_context_arena->occupied_bytes();
        out.host_context_reserved_bytes = host_context_arena->reserved_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace infernix::models::qwen3_5::detail
