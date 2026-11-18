#include "serve/server_tools.h"

#include <algorithm>
#include <utility>

namespace infernix::serve {

GenerationOutcome combine_round_outcomes(std::span<const ServerToolRound> rounds,
                                         GenerationOutcome final_round) {
    if (rounds.empty()) { return final_round; }
    GenerationOutcome total = std::move(final_round);
    std::vector<infernix::TokenId> generated;
    std::vector<infernix::SpeculativeStats::RoundStep> steps; // the earlier generations' rounds, in order
    const auto add_counts = [](std::vector<std::uint64_t>& into, const std::vector<std::uint64_t>& from) {
        if (into.size() < from.size()) { into.resize(from.size()); }
        for (std::size_t i = 0; i < from.size(); ++i) { into[i] += from[i]; }
    };
    for (const ServerToolRound& round : rounds) {
        const GenerationOutcome& earlier = round.outcome;
        total.prompt_tokens += earlier.prompt_tokens;
        total.completion_tokens += earlier.completion_tokens;
        total.reasoning_tokens += earlier.reasoning_tokens;
        generated.insert(generated.end(), earlier.generated_token_ids.begin(),
                         earlier.generated_token_ids.end());

        total.reused_prompt_tokens += earlier.reused_prompt_tokens;
        // Cache creation is one generation's own measurement past that generation's reused prefix
        // (infernix::GenerationResult), so every generation counts its own published creation once
        // and only the sum is the response's. A continuation that reused the whole earlier prefix
        // then holds the earlier creation inside its own reused prefix; after an eviction or a
        // failed reuse it does not: that round reports less reused, and the blocks it prefills
        // and publishes again are its own creation, so a position can count in two rounds, each
        // time for a creation that really ran. The optional stays unset when no generation
        // published creation.
        if (earlier.cache_created_prompt_tokens) {
            total.cache_created_prompt_tokens =
                total.cache_created_prompt_tokens.value_or(std::uint32_t{}) +
                *earlier.cache_created_prompt_tokens;
        }
        // Each generation's cached blocks are measured against its own reused prefix, so the
        // response's re-prefilled cached blocks are the summed blocks past the summed reuse.
        total.cached_block_tokens += earlier.cached_block_tokens;
        total.computed_prefill_tokens += earlier.computed_prefill_tokens;
        total.timings.prepare_seconds += earlier.timings.prepare_seconds;
        total.timings.vision_seconds += earlier.timings.vision_seconds;
        total.timings.prefill_seconds += earlier.timings.prefill_seconds;
        total.timings.decode_seconds += earlier.timings.decode_seconds;
        total.timings.decode_share_seconds += earlier.timings.decode_share_seconds;
        total.timings.prompt_wall_seconds += earlier.timings.prompt_wall_seconds;
        total.timings.generation_wall_seconds += earlier.timings.generation_wall_seconds;
        total.timings.total_seconds += earlier.timings.total_seconds;
        total.speculative.rounds += earlier.speculative.rounds;
        total.speculative.drafted_tokens += earlier.speculative.drafted_tokens;
        total.speculative.accepted_tokens += earlier.speculative.accepted_tokens;
        total.speculative.fallback_steps += earlier.speculative.fallback_steps;
        total.speculative.ngram_rounds += earlier.speculative.ngram_rounds;
        total.speculative.ngram_drafted_tokens += earlier.speculative.ngram_drafted_tokens;
        total.speculative.ngram_accepted_tokens += earlier.speculative.ngram_accepted_tokens;
        total.speculative.ngram_archive_rounds += earlier.speculative.ngram_archive_rounds;
        total.speculative.ngram_archive_drafted_tokens += earlier.speculative.ngram_archive_drafted_tokens;
        total.speculative.ngram_archive_accepted_tokens += earlier.speculative.ngram_archive_accepted_tokens;
        total.speculative.tree_rounds += earlier.speculative.tree_rounds;
        total.speculative.tree_side_rounds += earlier.speculative.tree_side_rounds;
        total.speculative.tree_side_accepted_tokens += earlier.speculative.tree_side_accepted_tokens;
        total.speculative.ngram_tree_rounds += earlier.speculative.ngram_tree_rounds;
        total.speculative.ngram_tree_side_rounds += earlier.speculative.ngram_tree_side_rounds;
        total.speculative.ngram_tree_side_accepted_tokens += earlier.speculative.ngram_tree_side_accepted_tokens;
        add_counts(total.speculative.accepted_per_position, earlier.speculative.accepted_per_position);
        add_counts(total.speculative.rounds_by_draft_length, earlier.speculative.rounds_by_draft_length);
        steps.insert(steps.end(), earlier.speculative.round_steps.begin(), earlier.speculative.round_steps.end());
        // Per-request counters (ExpertCacheStats), so the response's are the generations' sum.
        if (earlier.expert_cache) {
            infernix::ExpertCacheStats& cache = total.expert_cache.emplace(total.expert_cache.value_or(
                infernix::ExpertCacheStats{}));
            cache.routed += earlier.expert_cache->routed;
            cache.hits += earlier.expert_cache->hits;
            cache.decode_routed += earlier.expert_cache->decode_routed;
            cache.decode_hits += earlier.expert_cache->decode_hits;
        }
    }
    // The response's first token came from the first generation.
    total.timings.first_token_seconds = rounds.front().outcome.timings.first_token_seconds;
    generated.insert(generated.end(), total.generated_token_ids.begin(),
                     total.generated_token_ids.end());
    total.generated_token_ids = std::move(generated);
    steps.insert(steps.end(), total.speculative.round_steps.begin(), total.speculative.round_steps.end());
    total.speculative.round_steps = std::move(steps);
    return total;
}

void append_server_round(GenerationRequest& request, const ServerToolRound& round,
                         std::span<const std::string> server_tool_names) {
    const GenerationOutcome& outcome = round.outcome;
    ContentPart text;
    text.kind     = ContentKind::Text;
    text.type_raw = "text";
    text.text     = outcome.text;

    // An Assistant prefill was continued by the round's output: the call ends that same turn.
    const bool continued = request.continuation == infernix::PromptContinuationMode::ContinueFinalAssistant &&
                           !request.messages.empty() &&
                           request.messages.back().role == ChatRole::Assistant;
    if (continued) {
        ChatTurn& assistant = request.messages.back();
        assistant.reasoning_content += outcome.reasoning;
        if (!text.text.empty()) { assistant.content.push_back(std::move(text)); }
        assistant.tool_calls.insert(assistant.tool_calls.end(), round.calls.begin(),
                                    round.calls.end());
    } else {
        ChatTurn assistant;
        assistant.role              = ChatRole::Assistant;
        assistant.reasoning_content = outcome.reasoning;
        if (!text.text.empty()) { assistant.content.push_back(std::move(text)); }
        assistant.tool_calls = round.calls;
        request.messages.push_back(std::move(assistant));
    }
    request.messages.insert(request.messages.end(), round.results.begin(), round.results.end());
    request.continuation = infernix::PromptContinuationMode::NewAssistantTurn;
    request.max_tokens   = std::max(0, request.max_tokens - outcome.completion_tokens);

    ToolChoice& choice = request.tool_choice;
    if (choice.mode == ToolChoiceMode::Required) { choice.mode = ToolChoiceMode::Auto; }
    if (choice.allowed_names &&
        std::all_of(choice.allowed_names->begin(), choice.allowed_names->end(),
                    [&](const std::string& name) {
                        return std::find(server_tool_names.begin(), server_tool_names.end(),
                                         name) != server_tool_names.end();
                    })) {
        choice.allowed_names.reset();
    }
}

} // namespace infernix::serve
