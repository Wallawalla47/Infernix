#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5::detail {

namespace {

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label);

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

DecodeGraphProfile& DecodeGraphFamily::select(std::uint32_t batch_size, std::uint32_t frontier) {
    const auto it =
        std::find_if(profiles.begin(), profiles.end(), [&](const DecodeGraphProfile& p) {
            return p.batch_size == batch_size && p.min_execution_frontier <= frontier &&
                   frontier <= p.max_execution_frontier;
        });
    if (it == profiles.end()) {
        throw std::logic_error("CUDA Graph profile coverage is incomplete");
    }
    return *it;
}

DecodeGraphExecutable& DecodeGraphFamily::install(DecodeGraphProfile& profile) {
    const auto it =
        std::find_if(topologies.begin(), topologies.end(), [&](const DecodeGraphTopology& t) {
            return t.topology_class == profile.topology_class;
        });
    if (it == topologies.end()) { throw std::logic_error("CUDA Graph topology is unavailable"); }
    const auto index = static_cast<std::size_t>(&profile - profiles.data());
    if (it->installed_profile != index) {
        it->executable.update(profile.definition);
        it->installed_profile = index;
    }
    return it->executable;
}

void ProgramImpl::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    nvtx::ScopedRange prepare_range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);

    std::array<StateImageHandle, kMaximumConcurrency> capture_states{};
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        std::optional<StateImageHandle> state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        capture_states[row] = *state;
    }
    const auto capture_state_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };

    std::vector<KVAddressSpaceHandle> text_capture_allocations;
    std::vector<KVAddressSpaceHandle> mtp_capture_allocations;
    std::vector<KVAddressSpaceHandle> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_5::PagedKVCache& cache,
                                          KVAddressSpaceStore& addresses,
                                          std::vector<KVAddressSpaceHandle>& allocations,
                                          const char* label) {
        DeviceKVPagePool& pool       = cache.page_pool();
        KVExecutionTablePool& tables = cache.execution_tables();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            std::optional<KVAddressSpaceHandle> allocation =
                addresses.create_active(1, static_cast<std::int32_t>(row), device.stream);
            if (!allocation) { throw std::bad_alloc(); }
            allocations.push_back(*allocation);
            addresses.ensure_mapped_to_tokens(*allocation, 1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            tables.publish_repeated(addresses.execution_row(*allocation).handle(),
                                    addresses.physical_page(*allocation, 0),
                                    tables.logical_page_capacity(), device.stream);
        }
    };
    reserve_capture_rows(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                         "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), *backend_kv_addresses, mtp_capture_allocations,
                             "MTP KV cache");
    } else if (dflash && dflash->full) {
        reserve_capture_rows(*dflash->full, *backend_kv_addresses, dflash_capture_allocations,
                             "DFlash Full KV cache");
    }
    device.synchronize();

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->local_append_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages =
        [&](qwen3_5::PagedKVCache& cache, const KVAddressSpaceStore& addresses,
            const std::vector<KVAddressSpaceHandle>& allocations, std::uint32_t batch_size) {
            std::vector<DeviceKVPageHandle> pages;
            pages.reserve(batch_size);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                pages.push_back(addresses.physical_page(allocations[row], 0));
            }
            cache.page_pool().zero_pages(pages, device.stream);
        };
    // capture_tree selects a DFlash2 tree round: rows whose whole tree fits verify a tree of
    // capture_drafts + 1 columns built from capture_proposal_drafts main-chain proposals.
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size,
                                            std::uint32_t capture_drafts,
                                            std::uint32_t capture_proposal_drafts,
                                            bool capture_tree = false) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                           batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), *backend_kv_addresses,
                               mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(*dflash->full, *backend_kv_addresses, dflash_capture_allocations,
                               batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            state_images->zero_slot(capture_state_slot(row), device.stream);
            if (dflash) {
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress = {};
            *dflash_host_egress  = {};
            const std::uint32_t width = capture_drafts + 1U;
            const bool tree_row       = capture_tree && capacity - frontier - 1U >= capture_drafts;
            const std::uint32_t extent =
                tree_row ? capture_drafts
                         : std::min(capture_proposal_drafts, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                for (std::uint32_t step = 0; step < capture_drafts; ++step) {
                    const auto base = row * capture_drafts * ops::kSparseSpeculativeCandidates +
                                      step * ops::kSparseSpeculativeCandidates;
                    dflash_host_ingress->ngram_q[base] = 1.0F;
                    for (std::uint32_t slot = 0; slot < ops::kSparseSpeculativeCandidates; ++slot) {
                        dflash_host_ingress->ngram_candidates[base + slot] = slot;
                    }
                }
            }
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_valid_columns[row] =
                    static_cast<std::int32_t>((speculative_backend == SpeculativeBackend::DFlash2
                                                   ? capture_proposal_drafts
                                                   : std::min(extent, capture_proposal_drafts)) +
                                              1U);
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t column = 0; column < width; ++column) {
                    // The device adds a tree column's depth to the anchor position.
                    const std::uint32_t offset = tree_row ? 0U : std::min(column, extent);
                    dflash_host_ingress->target_rope_positions[row * width + column] = checked_i32(
                        frontier + offset, "graph representative DFlash target RoPE position");
                }
                dflash_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row]    = static_cast<std::int32_t>(row);
                dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(row);
                dflash_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                dflash_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                dflash_host_ingress->sampling[row]                = {};
                dflash_host_ingress->copy_rows[row]               = 1;
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(capture_drafts, capacity - frontier - 1U);
            const std::uint32_t width  = capture_drafts + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < capture_drafts; ++step) {
                    mtp_host_ingress->current_drafts[row * capture_drafts + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]       = static_cast<std::int32_t>(row);
                mtp_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                mtp_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                mtp_host_ingress->rope_deltas[row]             = 0;
                mtp_host_ingress->sampling[row]                = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->state_source_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->sampling[row]                = {};
            }
        }
    };
    const auto prepare_ordinary = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        prepare_representative(frontier, batch_size, 0, 0);
    };
    const auto execution_core = [&] {
        return execution::ExecutionCore{device,
                                        parameters,
                                        work,
                                        *state_images,
                                        replay_records ? &*replay_records : nullptr,
                                        io,
                                        prefill_hidden,
                                        prefill_chunk,
                                        proposal_head,
                                        prompt_attention};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        execution::OrdinaryBatchContext ordinary_state{
            execution_core(),      decoder->text_kv,
            *io.ordinary,          *ordinary_host_ingress,
            *ordinary_host_egress, state_images->continuation_hidden_store()};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_ordinary(code_warm.min, 1);
        device.synchronize();
        execution::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                         nullptr);
        device.synchronize();

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::CausalAttentionExecutionEnvelope envelope{planned.min + 1,
                                                                     planned.max + 1};
                execution::capture_ordinary_decode_batch(ordinary_state,
                                                         static_cast<std::int32_t>(batch_size),
                                                         envelope, profile.definition);
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device);
        for (auto& topology : ordinary_graphs.topologies) {
            const auto& profile = ordinary_graphs.profiles[*topology.installed_profile];
            prepare_ordinary(profile.min_execution_frontier, profile.batch_size);
            device.synchronize();
            topology.executable.launch(device.stream);
            device.synchronize();
        }
    }
    if (speculative_backend != SpeculativeBackend::None) {
        using execution::SpeculativePhase;
        const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
        // Every planned round family captures a Forward/Finish pair at its own width for every
        // batch size, on the one frame viewed at that width, and records ReplaySSM transitions
        // through the record view of the same width. Every MTP round proposes the next round's
        // drafts at the configured neural depth; a batch>1 DFlash ngram round also runs the
        // drafter so rows without a copy keep their neural proposal. A DFlash2 tree family serves
        // only the batch sizes that may verify its width. Chain families serve every batch size:
        // a round with a constrained row verifies the chain whatever the tree table selects.
        const std::uint32_t ar_depth = neural_draft_window;
        for (SpeculativeRoundFamily& round : round_families) {
            const bool ngram                  = round.shape.kind == SpeculativeRoundKind::Ngram;
            const bool tree                   = round.shape.kind == SpeculativeRoundKind::Tree;
            const std::uint32_t verify_drafts = round.shape.verify_drafts;
            const auto serves                 = [&](std::uint32_t batch) {
                return !tree || tree_widths.tree(batch, verify_drafts + 1U);
            };
            bool used = false;
            for (std::uint32_t batch = 1; batch <= max_concurrency; ++batch) {
                used = used || serves(batch);
            }
            if (!used) { continue; }
            // The DFlash drafter always proposes at its own width.
            const std::uint32_t proposal_drafts = mtp ? verify_drafts : neural_draft_window;
            const auto prepare = [&, verify_drafts, proposal_drafts, tree](std::uint32_t frontier,
                                                                           std::uint32_t batch) {
                prepare_representative(frontier, batch, verify_drafts, proposal_drafts, tree);
            };
            auto& forward_family = round.forward;
            auto& finish_family  = round.finish;
            const auto forward_profiles =
                mtp ? mtp_graph_profiles(capacity, verify_drafts, ar_depth)
                    : dflash_graph_profiles(speculative_backend, capacity, verify_drafts);
            // DFlash acceptance has no context-dependent nodes or resource envelope.
            const auto finish_profiles =
                mtp ? forward_profiles : std::vector<GraphExecutionProfile>{{0, capacity - 1}};
            validate_graph_profiles(forward_profiles, capacity - 1, "speculative forward");
            validate_graph_profiles(finish_profiles, capacity - 1, "speculative finish");
            const auto mtp_context = [&] {
                execution::ExecutionCore core = execution_core();
                core.replay_records           = round_replay_records(verify_drafts);
                execution::MtpBatchContext state{core,
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 state_images->continuation_hidden_store()};
                state.neural_proposal_drafts = ar_depth;
                return state;
            };
            const auto dflash_context = [&] {
                execution::ExecutionCore core = execution_core();
                core.replay_records           = round_replay_records(verify_drafts);
                execution::DFlashBatchContext state{core,
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    state_images->continuation_hidden_store(),
                                                    dflash_draft_handoff->tokens(),
                                                    dflash_draft_handoff->ready};
                state.ngram                  = ngram;
                state.neural_proposal_drafts = neural_draft_window;
                state.tree                   = tree;
                state.tree_paths             = draft_tree_paths;
                return state;
            };
            const auto target_envelope = [&](std::uint32_t frontier) {
                return ops::CausalAttentionExecutionEnvelope{
                    1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                           capacity, static_cast<std::uint64_t>(frontier) + verify_drafts + 1ULL))};
            };
            const auto mtp_envelopes = [&](std::uint32_t frontier) {
                return mtp_causal_attention_envelopes(frontier, verify_drafts, capacity, ar_depth);
            };
            const auto run = [&](SpeculativePhase phase, std::int32_t batch,
                                 std::uint32_t min_frontier, std::uint32_t max_frontier,
                                 DecodeGraphExecutable* executable) {
                if (mtp) {
                    auto state = mtp_context();
                    execution::mtp_decode_batch(state, batch, verify_drafts,
                                                mtp_envelopes(max_frontier), executable, phase);
                } else {
                    auto state = dflash_context();
                    execution::dflash_decode_batch(
                        state, batch, verify_drafts, dflash_envelopes(min_frontier, max_frontier),
                        target_envelope(max_frontier), executable, phase);
                }
            };
            const auto capture = [&](SpeculativePhase phase, DecodeGraphProfile& profile) {
                const auto batch = static_cast<std::int32_t>(profile.batch_size);
                const auto low   = profile.min_execution_frontier;
                const auto high  = profile.max_execution_frontier;
                if (mtp) {
                    auto state = mtp_context();
                    execution::capture_mtp_decode_batch(state, batch, verify_drafts,
                                                        mtp_envelopes(high), profile.definition,
                                                        phase);
                } else {
                    auto state = dflash_context();
                    execution::capture_dflash_decode_batch(
                        state, batch, verify_drafts, dflash_envelopes(low, high),
                        target_envelope(high), profile.definition, phase);
                }
            };

            std::uint32_t warm_batch = 1;
            while (!serves(warm_batch)) { ++warm_batch; }
            const auto code_warm = forward_profiles.front();
            prepare(code_warm.min, warm_batch);
            device.synchronize();
            run(SpeculativePhase::Forward, static_cast<std::int32_t>(warm_batch), code_warm.min,
                code_warm.max, nullptr);
            run(SpeculativePhase::Finish, static_cast<std::int32_t>(warm_batch), code_warm.min,
                code_warm.max, nullptr);
            device.synchronize();

            for (const auto phase : {SpeculativePhase::Forward, SpeculativePhase::Finish}) {
                const bool forward           = phase == SpeculativePhase::Forward;
                auto& family                 = forward ? forward_family : finish_family;
                const auto& planned_profiles = forward ? forward_profiles : finish_profiles;
                family.profiles.reserve(planned_profiles.size() * max_concurrency);
                for (std::uint32_t batch = 1; batch <= max_concurrency; ++batch) {
                    if (!serves(batch)) { continue; }
                    for (const auto planned : planned_profiles) {
                        family.profiles.emplace_back();
                        auto& profile                  = family.profiles.back();
                        profile.batch_size             = batch;
                        profile.min_execution_frontier = planned.min;
                        profile.max_execution_frontier = planned.max;
                        profile.topology_class =
                            planned.topology_class * max_concurrency + batch - 1U;
                        capture(phase, profile);
                    }
                }
                const char* label = forward ? (ngram  ? "ngram forward"
                                               : tree ? "tree forward"
                                                      : "speculative forward")
                                            : (ngram  ? "ngram finish"
                                               : tree ? "tree finish"
                                                      : "speculative finish");
                instantiate_graph_family(family, label, device);
            }

            // Warm complete transactions: finish consumes persistent outputs of its paired forward.
            for (auto& topology : forward_family.topologies) {
                auto& forward = forward_family.profiles[*topology.installed_profile];
                auto& finish =
                    finish_family.select(forward.batch_size, forward.min_execution_frontier);
                auto& finish_exec = finish_family.install(finish);
                prepare(forward.min_execution_frontier, forward.batch_size);
                device.synchronize();
                topology.executable.launch(device.stream);
                finish_exec.launch(device.stream);
                device.synchronize();
            }
        }
    }

    clear_stable_controls();
    state_images->zero_all(device.stream);
    if (dflash) {
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        if (!state_store->release(capture_states[row])) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }

    const auto release_capture_rows = [](KVAddressSpaceStore& addresses,
                                         std::vector<KVAddressSpaceHandle>& allocations) {
        for (const KVAddressSpaceHandle allocation : allocations) {
            addresses.deactivate(allocation);
            if (!addresses.release(allocation)) {
                throw std::logic_error("CUDA Graph capture KV address space could not be released");
            }
        }
        allocations.clear();
    };
    if (!dflash_capture_allocations.empty()) {
        release_capture_rows(*backend_kv_addresses, dflash_capture_allocations);
    }
    if (!mtp_capture_allocations.empty()) {
        release_capture_rows(*backend_kv_addresses, mtp_capture_allocations);
    }
    release_capture_rows(*text_kv_addresses, text_capture_allocations);
}


} // namespace infernix::models::qwen3_5::detail
