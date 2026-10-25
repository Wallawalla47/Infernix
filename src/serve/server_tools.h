#pragma once

// Protocol-neutral pieces of a server-side tool loop. When the model calls a tool the server
// executes itself (tool search), the response does not end there: the server appends the call and
// its result to the conversation and runs another generation, as a client would in its own tool
// loop. Every generation's prompt therefore extends the previous one's prompt and output, so the
// prefix cache serves each continuation, and a later request that replays the whole response as
// history renders the same tokens.

#include "serve/generation_service.h"
#include "serve/request.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace infernix::serve {

// Generations one response may run for server-side tools before it pauses (Anthropic
// `stop_reason: "pause_turn"`, Responses `incomplete`): the client resumes by sending the response
// back as history.
inline constexpr std::size_t kMaximumServerToolRounds = 10;

// One generation that ended in server-side tool calls, with what the server added after it.
struct ServerToolRound {
    GenerationOutcome outcome;
    // The generation's calls in call order, each with its wire identifier.
    std::vector<ToolCall> calls;
    // The result turns the server executed, in call order (client calls have none).
    std::vector<ChatTurn> results;
};

// The response totals of several generations: prompt, cache and output token counts, wall times
// and draft counts are summed; time to first token is the first generation's; everything else
// (output text, calls, finish reason, Engine identity) is the final generation's.
[[nodiscard]] GenerationOutcome combine_round_outcomes(std::span<const ServerToolRound> rounds,
                                                       GenerationOutcome final_round);

// Extends `request` with `round` — the assistant turn (reasoning, text and calls) followed by the
// result turns — so the next generation continues after them. The round's output tokens leave the
// response's budget, and a tool choice that forced a call was satisfied by it: the continuation's
// choice becomes automatic, and a selection that named only server tools is lifted.
void append_server_round(GenerationRequest& request, const ServerToolRound& round,
                         std::span<const std::string> server_tool_names);

} // namespace infernix::serve
