#include "models/qwen3_5/proposal_head.h"
#include "infernix/engine.h"
#include "graph_prepare_probe.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

infernix::EngineOptions engine_options(const char* artifact) {
    infernix::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = infernix::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.kv_cache                  = infernix::KvCacheStorage::Int8Group64;
    options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = infernix::test::available_proposal_head(artifact);
    options.enable_vision             = true;
    options.use_cuda_graph            = true;
    // The append check resumes from the decode-computed endpoint, which only free-cache-resume
    // publishes; the test asks for it, as the prefix tests do, whatever the artifact qualifies.
    options.fast_path_candidates      = {infernix::FastPath::FreeCacheResume};
    return options;
}

infernix::EngineOptions maximum_engine_options(const char* artifact) {
    infernix::EngineOptions options     = engine_options(artifact);
    options.max_context               = 262144;
    options.kv_capacity               = infernix::KvCapacityPolicy::explicit_capacity(262144);
    options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = infernix::test::available_proposal_head(artifact);
    return options;
}

infernix::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

int verify_loaded_product(const infernix::Engine& engine, bool graphs_prepared) {
    const infernix::LoadSummary load = engine.load_summary();
    if (load.architecture != "Qwen3_5MoeForCausalLM" || load.model_name.empty() ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "35B Engine construction has an invalid load summary: target="
                  << load.architecture << " weights=" << load.prefill_signature << '\n';
        return 1;
    }

    const infernix::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || vision->handoff_peak_bytes != 0 ||
        memory.workspace_logical_peak_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "35B Engine construction has an invalid memory summary\n";
        return 1;
    }
    if (!graphs_prepared || memory.cuda_graph_measured_bytes > memory.cuda_graph_allowance_bytes) {
        std::cerr << "35B Engine CUDA Graphs " << (graphs_prepared ? "prepared" : "not prepared")
                  << ", used " << memory.cuda_graph_measured_bytes << " bytes against an allowance of "
                  << memory.cuda_graph_allowance_bytes << '\n';
        return 1;
    }
    return 0;
}

int exercise_text_mtp_and_prefix(infernix::Engine& engine) {
    const std::vector<infernix::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    // The prefix cache publishes an endpoint only 64 or more tokens past the root
    // (prefix_cache::kMinimumTapSeparation), so the append check's first turn is a longer message.
    std::vector<infernix::TokenId> turn{248045, 846, 198};
    for (const infernix::TokenId token : engine.tokenize_text(
             "A lighthouse keeper writes down the weather every evening: the direction of the wind, "
             "the height of the waves, the colour of the sky at sunset and the ships that passed the "
             "point. After forty years the notebooks fill a shelf. Summarise what such a record could "
             "tell a historian about the coast, and name two questions it could not answer.")) {
        turn.push_back(token);
    }
    turn.insert(turn.end(), {248046, 198});
    if (turn.size() < 64) {
        std::cerr << "35B append fixture is too short for an endpoint: " << turn.size() << " tokens\n";
        return 1;
    }
    const infernix::GenerationResult first =
        engine.generate(engine.prepare_tokens(turn), greedy_options(5, true));
    if (first.generated_token_ids.size() != 5 ||
        first.speculative.backend != infernix::SpeculativeBackend::Mtp ||
        first.speculative.rounds == 0) {
        std::cerr << "35B fixed prompt did not complete through MTP: outputs="
                  << first.generated_token_ids.size() << '\n';
        return 1;
    }

    std::vector<infernix::TokenId> continuation = turn;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    const infernix::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(turn.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2) {
        std::cerr << "35B append/prefix reuse is incorrect: reused=" << reused.reused_prompt_tokens
                  << " expected=" << expected_reuse
                  << " outputs=" << reused.generated_token_ids.size() << '\n';
        return 1;
    }

    const auto stop_reference =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, false));
    if (stop_reference.generated_token_ids.size() != 5 ||
        stop_reference.generated_token_ids[0] == stop_reference.generated_token_ids[1]) {
        std::cerr << "35B partial-terminal fixture repeats its first token\n";
        return 1;
    }
    infernix::RequestOptions stop_options = greedy_options(5, false);
    stop_options.stop.token_ids.push_back(stop_reference.generated_token_ids[1]);
    const infernix::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), stop_options);
    if (stopped.finish_reason != infernix::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "35B custom stop did not terminate inside an MTP round\n";
        return 1;
    }

    std::vector<infernix::TokenId> stopped_continuation = prompt;
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(198);
    const infernix::GenerationResult after_stop = engine.generate(
        engine.prepare_tokens(std::move(stopped_continuation)), greedy_options(1, false));
    if (after_stop.generated_token_ids.size() != 1 ||
        after_stop.finish_reason != infernix::FinishReason::OutputLimit) {
        std::cerr << "35B request after partial MTP terminal did not finish its output budget\n";
        return 1;
    }

    return 0;
}

int exercise_vision(infernix::Engine& engine) {
    engine.reset_memory_peaks();
    const infernix::MemorySummary before = engine.memory_summary();
    if (!before.vision_workspace || before.vision_workspace->handoff_active_bytes != 0 ||
        before.vision_workspace->handoff_peak_bytes != 0) {
        std::cerr << "35B Vision handoff peak did not reset before Vision\n";
        return 1;
    }
    infernix::MessagePart image;
    image.kind              = infernix::MessagePartKind::Media;
    image.media.kind        = infernix::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(infernix::MessagePart{
        .kind = infernix::MessagePartKind::Text, .text = "What is visible?", .media = {}});

    infernix::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    const infernix::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), greedy_options(1, false));
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != infernix::FinishReason::OutputLimit) {
        std::cerr << "35B Vision request did not complete through the public Engine\n";
        return 1;
    }
    const infernix::MemorySummary after = engine.memory_summary();
    if (!after.vision_workspace ||
        after.vision_workspace->handoff_offset_bytes !=
            before.vision_workspace->handoff_offset_bytes ||
        after.vision_workspace->handoff_capacity_bytes !=
            before.vision_workspace->handoff_capacity_bytes ||
        after.vision_workspace->handoff_active_bytes != 0 ||
        after.vision_workspace->handoff_peak_bytes == 0 ||
        after.vision_workspace->handoff_peak_bytes >
            after.vision_workspace->handoff_capacity_bytes ||
        after.workspace_logical_peak_bytes == 0 ||
        after.workspace_logical_peak_bytes > after.workspace.capacity_bytes) {
        std::cerr << "35B Vision request did not use the startup-frozen workspace handoff\n";
        return 1;
    }
    return 0;
}

int exercise_maximum_configuration(const char* artifact) {
    infernix::Engine engine(maximum_engine_options(artifact));
    const infernix::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.max_context != 262144 || memory.kv_cache != infernix::KvCacheStorage::Int8Group64 ||
        memory.kv_payload_bytes == 0 || memory.sequence.capacity_bytes == 0 ||
        memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "35B maximum configuration does not match the planned 256K layout: context="
                  << memory.max_context << " kv_payload=" << memory.kv_payload_bytes
                  << " sequence=" << memory.sequence.capacity_bytes
                  << " workspace=" << memory.workspace.capacity_bytes << '\n';
        return 1;
    }

    std::vector<infernix::TokenId> oversized(262145, 198);
    bool rejected = false;
    try {
        (void)engine.generate(engine.prepare_tokens(std::move(oversized)),
                              greedy_options(1, false));
    } catch (const std::invalid_argument&) { rejected = true; } catch (const std::out_of_range&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "35B maximum configuration accepted an over-capacity request\n";
        return 1;
    }

    const std::vector<infernix::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const infernix::GenerationResult probe =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(1, false));
    if (probe.generated_token_ids.size() != 1) {
        std::cerr << "35B Engine was unusable after rejecting an over-capacity request\n";
        return 1;
    }
    const infernix::MemorySummary after = engine.memory_summary();
    if (after.workspace_logical_peak_bytes == 0 ||
        after.workspace_logical_peak_bytes > after.workspace.capacity_bytes) {
        std::cerr << "35B maximum configuration did not report its executed workspace phase\n";
        return 1;
    }

    // Offloaded experts (122B-A10B): the text KV is elastic. Its first 32,768 tokens are backed at
    // startup, so a 40,960-token prompt runs only if the pool grows from the expert frames; run twice
    // (the second after the first's pages were released), the greedy outputs must agree.
    if (const auto& cache = memory.expert_cache; cache && cache->kv_capacity_pages != 0) {
        if (cache->kv_backed_pages != 512 || cache->kv_capacity_pages < 4096 || cache->kv_backed_bytes == 0) {
            std::cerr << "122B elastic KV did not start at its 32,768-token base: " << cache->kv_backed_pages << " of "
                      << cache->kv_capacity_pages << " pages backed\n";
            return 1;
        }
        const std::vector<infernix::TokenId> sentence = engine.tokenize_text(
            "The tide tables of the harbour list every high and low water of the year, and the keeper "
            "checks them against the marks on the pier each morning. ");
        std::vector<infernix::TokenId> long_prompt{248045, 846, 198};
        while (long_prompt.size() < 40960 - 2) {
            long_prompt.insert(long_prompt.end(), sentence.begin(), sentence.end());
        }
        long_prompt.resize(40960 - 2);
        long_prompt.insert(long_prompt.end(), {248046, 198});
        const auto run = [&] {
            return engine.generate(engine.prepare_tokens(long_prompt), greedy_options(4, false)).generated_token_ids;
        };
        const auto short_run = [&] {
            return engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false)).generated_token_ids;
        };
        const std::vector<infernix::TokenId> short_before = short_run();
        const std::vector<infernix::TokenId> grown        = run();
        const auto between                                = engine.memory_summary().expert_cache;
        // After the release shrank the pool back to its base, the KV table row the long request used
        // still names its pages past a short request's own, the top ones now unmapped (candidates
        // OR-14). A short request on that row must neither fault (a read of an unmapped page past its
        // length) nor change its bits (a read of a stale mapped one) against the same request before
        // the pool grew.
        if (between->kv_backed_pages != 512) {
            std::cerr << "122B elastic KV: the long request's release left " << between->kv_backed_pages
                      << " pages backed, not the base 512, so the post-shrink table check did not run\n";
            return 1;
        }
        const std::vector<infernix::TokenId> short_after = short_run();
        if (short_after.size() != 16 || short_after != short_before) {
            std::cerr << "122B elastic KV: a short request after the shrink differs from the same request before the "
                         "growth (a kernel read table entries past its row's pages)\n";
            return 1;
        }
        const std::vector<infernix::TokenId> again = run();
        const auto final_cache                      = engine.memory_summary().expert_cache;
        std::cout << "elastic KV: " << between->kv_backed_pages << " pages backed after the first long request, "
                  << final_cache->kv_backed_pages << " after the second (base 512, capacity "
                  << final_cache->kv_capacity_pages << "); expert frames " << final_cache->frames << '\n';
        if (grown.size() != 4 || grown != again || final_cache->kv_backed_pages > final_cache->kv_capacity_pages) {
            std::cerr << "122B elastic KV: the long prompt's outputs differ between the grown pool and the regrown "
                         "one, or the summary is inconsistent\n";
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }

    try {
        {
            const infernix::test::GraphPrepareProbe graphs;
            infernix::Engine engine(graphs.attached(engine_options(artifact)));
            if (const int result = verify_loaded_product(engine, graphs.prepared()); result != 0) { return result; }
            if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) { return result; }
            if (const int result = exercise_vision(engine); result != 0) { return result; }
        }
        if (const int result = exercise_maximum_configuration(artifact); result != 0) { return result; }
    } catch (const std::exception& error) {
        std::cerr << "real MoE Engine test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
