#include "serve/server_tools.h"

#include <algorithm>
#include <utility>

namespace infernix::serve {

GenerationOutcome combine_round_outcomes(std::span<const ServerToolRound> rounds,
                                         GenerationOutcome final_round) {
    if (rounds.empty()) { return final_round; }
    GenerationOutcome total = std::move(final_round);
    std::vector<infernix::TokenId> generated;
    for (const ServerToolRound& round : rounds) {
        const GenerationOutcome& earlier = round.outcome;
        total.prompt_tokens += earlier.prompt_tokens;
        total.completion_tokens += earlier.completion_tokens;
        total.reasoning_tokens += earlier.reasoning_tokens;
        generated.insert(generated.end(), earlier.generated_token_ids.begin(),
                         earlier.generated_token_ids.end());

        total.reused_prompt_tokens += earlier.reused_prompt_tokens;
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
    }
    // The response's first token came from the first generation.
    total.timings.first_token_seconds = rounds.front().outcome.timings.first_token_seconds;
    generated.insert(generated.end(), total.generated_token_ids.begin(),
                     total.generated_token_ids.end());
    total.generated_token_ids = std::move(generated);
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
