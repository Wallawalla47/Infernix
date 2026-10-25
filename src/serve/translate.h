#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "infernix/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<infernix::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<infernix::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

infernix::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// The text a turn that loads tools ends with: the definitions in the function form of the listed
// tools, framed like the prompt's tool section. It depends on the definitions alone, so a history
// replayed by a later request renders the same tokens.
std::string render_loaded_tools(std::span<const ToolDefinition> tools);

// Each declaration's protocol schema path in Engine declaration order (listed tools, then deferred
// ones), for mapping an Engine tool-schema error back to the request.
std::vector<std::string> tool_declaration_params(const GenerationRequest& req);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
infernix::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

// The output tokens a request can emit: its requested limit, or the room its prompt leaves in
// max_context when that is less (the last output token needs no context position). Engine
// enforces the same limit; 0 for a prompt past max_context, which Engine rejects.
std::uint32_t effective_output_limit(std::uint32_t requested_output_tokens,
                                     std::uint32_t max_context, std::uint32_t prompt_tokens);

// The thinking budget ServeOptions::thinking_answer_room gives a thinking request without one of
// its own: thinking closes early enough to leave the answer a quarter of output_limit (at least
// kThinkingAnswerRoomMinReserve, at most half), from kThinkingAnswerRoomMinTokens. output_limit is
// the effective limit, not max_tokens: a prompt that leaves less room than max_tokens still gets
// its answer room. Empty below the threshold or with the room off.
std::optional<std::uint32_t> thinking_answer_room_budget(const ServeOptions& server,
                                                         std::uint32_t output_limit);

} // namespace infernix::serve
