#pragma once

// Inkling 2's constrained output (docs/maintainer/constrained-decoding.md §4.4): the grammar compiler over its
// vocabulary, and the grammar of a turn under a content constraint, a tool policy, or both.
//
// A turn is blocks (output_session.h): `<|message_model|>NAME?<|content_K|>BODY<|end_message|>`, then the model's stop
// token <|content_model_end_sampling|>. The grammars name the seven markers (OutputMarkers) by token. The compiler
// spells each 0xFF, a byte no UTF-8 rule matches (GBNF literals and classes, regex, JSON), so a block's body never
// holds a marker and only a token-level rule writes one; every other special token has no spelling and no rule admits
// it.
//
// - thinking block: `<|message_model|><|content_thinking|>` free text `<|end_message|>`, free text being any token but a
//   marker (a thinking budget's control, its message text then <|end_message|>, ends one);
// - answer block (a content constraint): `<|message_model|><|content_text|>` the constraint's text `<|end_message|>`,
//   after any thinking blocks; the turn then ends;
// - text block (a tool policy's free answer): `<|message_model|>` a content opener (text, tool_error, invoke_tool_text),
//   free text, `<|end_message|>`;
// - call block: `<|message_model|>NAME?<|content_invoke_tool_json|>{"name": "NAME", "args": ARGS}<|end_message|>`, one
//   alternative per callable tool (a header's name, when written, is the call's, as the chat template writes it). ARGS
//   is the tool's parameter schema when the tool is strict, else any JSON object. The call's JSON takes whitespace in
//   runs of at most 32, as a JSON answer does (text/json_schema.cpp).
//
// Tool policies (Qwen's contract, constrained-decoding.md §4.3): auto lets thinking and text blocks come first and ends
// with the calls, if any; required and named choices allow thinking blocks, then calls; none allows no call; parallel
// false allows one call. With a JSON content constraint, auto allows the answer or the calls, required the calls.

#include "infernix/types.h"
#include "models/inkling2/frontend/output_session.h"
#include "text/grammar.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infernix::text {
class Tokenizer;
}

namespace infernix::models::inkling2 {

// The markers in the grammar's reserved order.
[[nodiscard]] std::vector<TokenId> grammar_markers(const OutputMarkers& markers);

// The compiler over `tokenizer`'s vocabulary: each ordinary token's bytes, each marker 0xFF (reserved: a request's GBNF
// may not name one), every other special token none; `eos` the model's stop tokens.
[[nodiscard]] std::unique_ptr<text::GrammarCompiler> make_grammar_compiler(const text::Tokenizer& tokenizer,
                                                                         const OutputMarkers& markers,
                                                                         std::vector<TokenId> eos,
                                                                         std::size_t cache_bytes);

// The callable tools of one turn's constrained tool route.
struct ToolPolicy {
    std::vector<DeclaredTool> tools; // empty: the turn may call none
    bool required = false;
    bool parallel = true;
};

// The tool route `choice` selects over `declared` (null: the prompt declares none): nullopt when the turn's calls are
// free (an auto choice of non-strict tools under ToolConstraintMode::Automatic, or no tools); else the constrained
// policy. Throws RequestError (InvalidToolConstraint) for a choice that names an undeclared tool, or a required choice
// with no callable tool.
[[nodiscard]] std::optional<ToolPolicy> select_tool_policy(const ToolNames* declared, const ToolChoice& choice);

// The turn's matcher: `content` inside the turn's framing, `tools` (with callable tools) as calls, or both (`content`
// then JSON). Throws RequestError for an invalid constraint or tool schema.
[[nodiscard]] std::unique_ptr<text::GrammarSession> compile_turn_grammar(text::GrammarCompiler& compiler,
                                                                       const OutputMarkers& markers,
                                                                       const std::optional<OutputConstraint>& content,
                                                                       const ToolPolicy* tools,
                                                                       text::GrammarCompiler::Compile mode);

} // namespace infernix::models::inkling2
