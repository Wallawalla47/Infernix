#include "infernix/engine.h"

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
    options.speculative.proposal_head = infernix::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.use_cuda_graph            = true;
    return options;
}

infernix::EngineOptions maximum_engine_options(const char* artifact) {
    infernix::EngineOptions options     = engine_options(artifact);
    options.max_context               = 262144;
    options.kv_capacity               = infernix::KvCapacityPolicy::explicit_capacity(262144);
    options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = infernix::ProposalHead::Optimized;
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

int verify_loaded_product(const infernix::Engine& engine) {
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
    if (memory.cuda_graph_measured_bytes == 0 ||
        memory.cuda_graph_measured_bytes > memory.cuda_graph_allowance_bytes) {
        std::cerr << "35B Engine CUDA Graphs used " << memory.cuda_graph_measured_bytes
                  << " bytes against an allowance of " << memory.cuda_graph_allowance_bytes
                  << '\n';
        return 1;
    }
    return 0;
}

int exercise_text_mtp_and_prefix(infernix::Engine& engine) {
    const std::vector<infernix::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const infernix::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5, true));
    if (first.generated_token_ids.size() != 5 ||
        first.speculative.backend != infernix::SpeculativeBackend::Mtp ||
        first.speculative.rounds == 0) {
        std::cerr << "35B fixed prompt did not complete through MTP: outputs="
                  << first.generated_token_ids.size() << '\n';
        return 1;
    }

    std::vector<infernix::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    const infernix::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
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
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }

    {
        infernix::Engine engine(engine_options(artifact));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_text_mtp_and_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_maximum_configuration(artifact); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
