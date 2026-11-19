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
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {

// Generations one response may run for server-side tools before it pauses (Anthropic
// `stop_reason: "pause_turn"`; a Response ends `completed` after its last searches): the client
// resumes by sending the response back as history.
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

// Why a server-side tool loop's last generation ends the response: it answered or called a client tool, or it was
// not a turn's end (Finished); the round limit or the protocol's own limit stopped the loop (RoundLimit); the
// continuation had no output tokens left (OutputLimit) or no room in the context (ContextWindow).
enum class ServerRoundsEnd { Finished, RoundLimit, OutputLimit, ContextWindow };

// One generation's calls as its protocol classified them: each call with its wire identifier, the result turns of
// those the server executed (in call order), and whether any call is the client's.
struct ServerRoundCalls {
    std::vector<ToolCall> calls;
    std::vector<ChatTurn> results;
    bool client_calls = false;
};

// Runs `prepared` and every continuation its server-side tool calls lead to: a generation that called only server
// tools and ended its turn has its round appended to `continuation`, and the next generation runs, until one answers
// or calls a client tool, or a limit ends the response. `prepared` ends holding the last generation's request (and
// its ingress reservation); `rounds` receives each continued round. The protocol supplies
//   ServerRoundCalls calls(const GenerationOutcome&)  classifies the calls and executes the server's;
//   bool limit_reached()                               a limit of its own, read after calls();
//   void keep()                                        a continued round, after it joined `rounds`;
//   void end(ServerToolRound&&, ServerRoundsEnd)       the final generation and why the response ends.
template <class Protocol>
void run_server_tool_rounds(GenerationBackend& service, PreparedRequest& prepared, const StreamSink* sink,
                            const std::function<bool()>& is_cancelled,
                            const std::function<PreparedRequest(const GenerationRequest&)>& prepare_next,
                            GenerationRequest& continuation, std::vector<ServerToolRound>& rounds,
                            std::span<const std::string> server_tool_names, Protocol& protocol) {
    for (;;) {
        GenerationOutcome outcome = service.run(prepared, sink, is_cancelled);
        ServerRoundCalls calls    = protocol.calls(outcome);
        // A generation continues only when its turn ended in server calls alone; a server call in a turn that also
        // calls a client tool, or that the output limit cut, still runs, so the history the client sends back
        // carries its result.
        const bool turn_ended = outcome.finish_reason == infernix::FinishReason::StopToken ||
                                outcome.finish_reason == infernix::FinishReason::None;
        bool continues      = !calls.results.empty() && !calls.client_calls && turn_ended;
        ServerRoundsEnd end = ServerRoundsEnd::Finished;
        if (continues && (rounds.size() + 1U >= kMaximumServerToolRounds || protocol.limit_reached())) {
            continues = false;
            end       = ServerRoundsEnd::RoundLimit;
        }
        ServerToolRound round{.outcome = std::move(outcome),
                              .calls   = std::move(calls.calls),
                              .results = std::move(calls.results)};
        std::optional<PreparedRequest> next;
        if (continues) {
            GenerationRequest extended = continuation;
            append_server_round(extended, round, server_tool_names);
            if (extended.max_tokens <= 0) {
                continues = false;
                end       = ServerRoundsEnd::OutputLimit;
            } else {
                try {
                    next.emplace(prepare_next(extended));
                    continuation = std::move(extended);
                } catch (const ApiException& error) {
                    if (error.error().code != "context_length_exceeded") { throw; }
                    continues = false;
                    end       = ServerRoundsEnd::ContextWindow;
                }
            }
        }
        if (!continues) {
            protocol.end(std::move(round), end);
            return;
        }
        rounds.push_back(std::move(round));
        protocol.keep();
        prepared = std::move(*next);
    }
}

} // namespace infernix::serve
