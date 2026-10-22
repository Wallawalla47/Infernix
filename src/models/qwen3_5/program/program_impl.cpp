#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/lossless.h"
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
                         DeviceContext& device_in, const StartupObserver& startup_observer,
                         const ProgramOffloadOptions* offload_options)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      idle_prefill_width(plan.idle_prefill_width),
      execution_schedule(plan.execution_schedule), draft_window(plan.draft_window),
      neural_draft_window(plan.neural_draft_window), tree_widths(plan.tree_widths),
      draft_tree_paths(plan.draft_tree_paths), ngram_draft_window(plan.ngram_draft_window),
      ngram_batch_window(plan.ngram_batch_window != 0 ? plan.ngram_batch_window : plan.ngram_draft_window),
      ngram_min_match(plan.ngram_min_match), speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::PrefillRoundHost))),
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
    if (execution_schedule.mtp_draft_confidence) {
        // A verified draft column's prior cost against a round without drafts: with experts held
        // outside device memory (the only configuration the schedule enables this for) its misses,
        // Qwen3.5-122B design §0: ~10 ms a column against 13.8 ms a plain token on the 5090. The fit
        // replaces it.
        mtp_confidence.emplace(0.7);
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
    if (speculative_backend == SpeculativeBackend::Mtp &&
        std::count_if(plan.round_shapes.begin(), plan.round_shapes.end(),
                      [](const SpeculativeRoundShape& shape) {
                          return shape.kind == SpeculativeRoundKind::Neural;
                      }) > 1) {
        // adaptive-mtp: one neural family per draft length, serving the batch sizes the policy
        // may run it at.
        mtp_draft_policy.emplace(neural_draft_window, max_concurrency);
        for (std::uint32_t k = 1; k <= neural_draft_window; ++k) {
            const bool planned = std::any_of(
                plan.round_shapes.begin(), plan.round_shapes.end(),
                [&](const SpeculativeRoundShape& shape) {
                    return shape.kind == SpeculativeRoundKind::Neural && shape.verify_drafts == k &&
                           shape.single_row == (max_concurrency > 1 &&
                                                !mtp_draft_policy->serves(k, max_concurrency));
                });
            if (!planned) {
                throw std::logic_error("adaptive MTP draft lengths do not match the planned families");
            }
        }
    }
    if (std::any_of(plan.round_shapes.begin(), plan.round_shapes.end(),
                    [](const SpeculativeRoundShape& shape) { return shape.split; })) {
        // adaptive-dflash: one split family per width the policy may choose, serving the batch
        // sizes it may run that width at. With a batch-only plain family, rounds of several rows
        // run it and the policy chooses for one row only.
        for (const SpeculativeRoundShape& shape : plan.round_shapes) {
            if (shape.batch_only) { dflash_batch_drafts = shape.verify_drafts; }
        }
        // With offloaded experts a round may verify no draft (width 1: the narrowest family with
        // every row's drafts masked, which routes no expert for them).
        dflash_width_policy.emplace(neural_draft_window, max_concurrency, plan.dflash_width_costs,
                                    dflash_batch_drafts == 0, offload_options != nullptr);
        for (std::uint32_t batch = 1; batch <= max_concurrency; ++batch) {
            for (const std::uint32_t width : dflash_width_policy->widths(batch)) {
                if (width == 1U) { continue; } // the narrowest family serves it
                const bool planned = std::any_of(
                    plan.round_shapes.begin(), plan.round_shapes.end(), [&](const SpeculativeRoundShape& shape) {
                        return shape.split && shape.verify_drafts + 1U == width && (!shape.single_row || batch == 1U);
                    });
                if (!planned) {
                    throw std::logic_error("adaptive DFlash widths do not match the planned families");
                }
            }
        }
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    // Elastic text KV: the page planes over a reserved region, its base backed now (elastic_kv.cpp).
    DeviceSpan text_pages{};
    if (plan.persistent.text_kv_region_bytes != 0) {
        kv_layout     = plan.persistent.decoder.text_kv.pages;
        kv_base_pages = plan.persistent.text_kv_base_pages;
        kv_vmm        = std::make_unique<VmmRange>(device.device, plan.persistent.text_kv_region_bytes, kKvChunkBytes);
        if (!map_kv(kv_base_pages)) {
            throw std::runtime_error("the text KV's first " + std::to_string(kv_base_pages) +
                                     " pages do not fit in device memory");
        }
        text_pages = DeviceSpan{kv_vmm->base(), plan.persistent.text_kv_region_bytes};
    }
    decoder      = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder, text_pages);
    if (kv_vmm) { decoder->text_kv.page_pool().set_backed_pages(kv_base_pages); }
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
    // Device state and KV only: the prefix cache owns the whole Host budget as its slab pool
    // (hybrid_program.cpp). Every lane owns one KV address space per pool, and a binding stages
    // the next one before its lane is published.
    state_store   = std::make_unique<StateImageStore>(*state_images);
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), decoder->text_kv.page_pool().capacity_pages());
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), max_concurrency + 1U,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), backend->page_pool().capacity_pages());
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), max_concurrency + 1U,
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
        CUDA_CHECK(cudaMemsetAsync(io.constraint->observe.data, 0xff,
                                   io.constraint->observe.bytes(), device.stream));
        constraint_descriptors_host_.assign(kTokenConstraintColumns, -1);
        constraint_observe_host_.assign(
            static_cast<std::size_t>(ops::kTokenObserveTokens) * kTokenConstraintColumns, -1);
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
    if (plan.persistent.grammar_masks) {
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_masks_device.bytes());
    }
    if (is_masked_draft_backend(speculative_backend)) {
        dflash_draft_handoff.emplace(device, draft_window * max_concurrency);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    host_tokens = round_host
                      ? &static_cast<qwen3_5::PrefillRoundHost*>(round_host->data())->sampled_token
                      : nullptr;
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
    if (offload_options != nullptr) {
        // Offloaded experts (design §9.2): the engine after every fixed allocation, its frames sized
        // from what they leave less the graphs' allowance, started before the graphs capture the
        // experts' sources.
        const OffloadGeometry geometry{
            .lanes       = static_cast<std::int32_t>(max_concurrency),
            .max_width   = static_cast<std::int32_t>(draft_window + 1U),
            .columns     = static_cast<std::int32_t>(std::max<std::uint32_t>(
                std::max(idle_prefill_width, prefill_chunk), max_concurrency * (draft_window + 1U))),
            .chunk       = static_cast<std::int32_t>(prefill_chunk),
            .speculative = speculative_backend != SpeculativeBackend::None || ngram_draft_window != 0,
            .graph_bound = static_cast<std::uint32_t>(
                (graph_allowance_bytes + runtime::expert_offload::kGraphExecutableBytes - 1) /
                runtime::expert_offload::kGraphExecutableBytes),
            .wide_pass_bytes = std::max(workspace_plan.text_prefill_wide, workspace_plan.mtp_prefill_wide),
            .device      = device.device};
        offload = std::make_unique<ProgramOffload>(parameters, geometry, offload_options->engine,
                                                   offload_options->diagnostic);
        offload->start(device.stream);
        io.offloaded_experts = &offload->experts();
        // The elastic text KV's evicting shrink runs in an idle engine's maintenance, which the cache's
        // monitor requests once the pool has been idle long enough (elastic_kv.cpp).
        if (kv_vmm) {
            offload->engine().set_maintenance_due([this] { return kv_idle_due(); });
            offload->engine().report("elastic text KV: " + std::to_string(kv_base_pages) + " of " +
                                     std::to_string(decoder->text_kv.page_pool().capacity_pages()) +
                                     " pages backed at startup (" + std::to_string(kv_vmm->mapped_bytes() >> 20) +
                                     " MiB); the rest grows from the expert frames as requests need it");
        }
        device.synchronize();
    }
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        const std::size_t free_before = device.free_bytes();
        // Offloaded experts: each executable's warm launch runs the MoE (its CPU-served misses and
        // landings included), so a round is open; an expert the warm launches could not serve fails
        // the startup.
        if (offload) { offload->begin_round(device.stream); }
        prepare_graphs();
        device.synchronize();
        if (offload) { offload->engine().check_error(parameters.model.info().name + " (graph preparation)"); }
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
            // The lossless route scores from the correctly rounded FP32 logits.
            const bool lossless = execution_schedule.lossless;
            Tensor logits       = work.alloc(lossless ? DType::FP32 : DType::BF16,
                                             {dimension(parameters.model.config().text.vocab_size), columns});
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
            if (lossless) {
                execution::lossless_head(hidden, parameters.text.output_head.weight, logits, work, device.stream);
            } else {
                execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            }
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
                 prefill_chunk, proposal_head, execution_schedule},
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
            // Offloaded experts: each scored pass is a prompt pass of lane 0.
            execution::OffloadedCall offload_call;
            if (offload) {
                offload_call = offload->begin_pass(device.stream, static_cast<std::int32_t>(nominal));
                schedule_state.offloaded_call = offload_call;
            }
            execution::PrefillChunkResult result;
            try {
                result = execution::prefill_text_chunk(schedule_state, std::span<const TokenId>(prompt.token_ids),
                                                       nominal, std::nullopt, false);
            } catch (...) {
                if (offload) { offload->abandon_pass(offload_call); }
                throw;
            }
            if (offload) {
                const bool last = cursor + result.processed_tokens == predictor_count;
                offload->end_pass(device.stream, offload_call, ProgramOffload::PassKind::Prompt, 0,
                                  static_cast<std::int32_t>(result.processed_tokens), cursor, last,
                                  parameters.model.info().name);
                if (last) { offload->return_stream(device.stream); }
            }
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
    if (offload) {
        out.expert_cache = offload->engine().memory_summary();
        if (kv_vmm) {
            const DeviceKVPagePool& pool        = decoder->text_kv.page_pool();
            out.expert_cache->kv_backed_pages   = pool.backed_pages();
            out.expert_cache->kv_capacity_pages = pool.capacity_pages();
            out.expert_cache->kv_backed_bytes   = kv_vmm->mapped_bytes();
        }
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
