#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"
#include "text/grammar.h"

namespace infernix::models::qwen3_5::frontend {
// Every call opens with the model's `<tool_call>` token (`tool_call_token`); free text may spell
// the marker with ordinary tokens and stays text, as the output decoder reads it. `continuation`
// is the assistant prefix as tokens, its `<tool_call>` spellings as that token.
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolCallOutputContract& contract,
                     std::int32_t tool_call_token, const text::ReasoningFraming& reasoning,
                     std::span<const std::int32_t> continuation,
                     const std::optional<OutputConstraint>& body = {});
} // namespace infernix::models::qwen3_5::frontend
