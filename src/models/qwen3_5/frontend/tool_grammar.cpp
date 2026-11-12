#include "models/qwen3_5/frontend/tool_grammar.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/json_schema.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <optional>
#include <string_view>
#include <vector>

namespace infernix::models::qwen3_5::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

constexpr std::string_view kParameterOpen = "<parameter=";
constexpr std::string_view kValueEnd      = "\n</parameter>";
constexpr std::string_view kCallClose     = "</function>\n</tool_call>";

// The rules of the constrained tool route the grammar cannot state. `names`: a call names each
// parameter once. Non-strict names are open (the grammar orders strict ones already), so a streamed
// call could otherwise repeat a key it has published. `quotes`: in the answer's free text, a
// `<tool_call>` token inside Markdown code is a quoted example (MarkdownCodeTracker, as the parser
// reads it), so the matcher does not see it and the text goes on; elsewhere it opens a call.
// Masking the token there instead would change what the model writes. The companion reads only the
// canonical framing the grammar emits: `<tool_call>` (the token) `\n<function=NAME>\n`, then
// `<parameter=NAME>\nVALUE\n</parameter>\n` per parameter, then `</function>\n</tool_call>`, and `\n`
// before the next call's token. A value ends at its first `\n</parameter>` (a JSON value cannot
// spell it), except that a raw value goes on when text other than the next call's token follows the
// whole call's closing lines (the grammar's raw string rule). `calls`: the grammar's raw string reads
// the `<tool_call>` token right after those closing lines (or after their `\n` when calls are not
// parallel) as value text, while the decoder takes the token there only as the next call's opener,
// so it is masked there; every tool grammar has this rule. `declared` (Contract::declared_names): a
// raw value's `\n</parameter>\n<parameter=NAME>` opens NAME only when the call declares it and has
// not named it yet; any other NAME makes the line the value's text (the grammar keeps both readings).
class ToolCallRules final : public text::GrammarStateConstraint {
public:
    using DeclaredNames = std::vector<std::pair<std::string, std::vector<std::string>>>;

    ToolCallRules(std::shared_ptr<const std::vector<std::string>> vocabulary,
                  std::shared_ptr<const std::vector<std::int32_t>> tag_ends, std::int32_t tool_call,
                  const text::ReasoningFraming& reasoning, bool combined, bool parallel, bool names,
                  bool quotes, std::shared_ptr<const DeclaredNames> declared)
        : vocabulary_(std::move(vocabulary)), tag_ends_(std::move(tag_ends)), tool_call_(tool_call),
          reasoning_close_(reasoning.close_token), separator_(reasoning.separator.size()),
          body_phase_(combined ? Phase::Branch : Phase::Text), parallel_(parallel), names_rule_(names),
          quotes_rule_(quotes), declared_(std::move(declared)) {
        cursor_.phase = reasoning_close_ ? Phase::Reasoning : body_phase_;
        committed_    = {cursor_, names_};
    }

    void filter(std::span<std::uint32_t> mask) override {
        if (cursor_.phase == Phase::AfterCall && !opens_call(cursor_)) {
            mask[static_cast<std::size_t>(tool_call_) / 32] &= ~(1u << (static_cast<std::uint32_t>(tool_call_) % 32));
        }
        if (!names_rule_ || !in_call(cursor_.phase)) { return; }
        // Only a token holding '>' completes a name; outside a name or a started opener it must also
        // hold the whole opener. Value tokens need no simulation.
        const bool any = cursor_.phase == Phase::Name || (cursor_.phase == Phase::Parameters && cursor_.matched != 0);
        for (const std::int32_t id : *tag_ends_) {
            std::uint32_t& word = mask[static_cast<std::size_t>(id) / 32];
            const std::uint32_t bit = 1u << (static_cast<std::uint32_t>(id) % 32);
            if ((word & bit) == 0) { continue; }
            const std::string& bytes = (*vocabulary_)[static_cast<std::size_t>(id)];
            if (!any && bytes.find(kParameterOpen) == std::string::npos) { continue; }
            Cursor cursor = cursor_;
            Names names{&names_};
            if (!step(cursor, names, id)) { word &= ~bit; }
        }
    }

    [[nodiscard]] bool literal(std::int32_t token) const noexcept override {
        return quoted(cursor_, token);
    }

    bool accept(std::int32_t token) override {
        Cursor cursor = cursor_;
        Names names{&names_};
        if (!step(cursor, names, token)) { return false; }
        cursor_ = cursor;
        names.apply(names_);
        return true;
    }

    void confirm() noexcept override { committed_ = {cursor_, names_}; }

    void discard() noexcept override {
        cursor_ = committed_.cursor;
        names_  = committed_.names;
    }

private:
    enum class Phase : std::uint8_t {
        Reasoning,  // before the reasoning close token
        Separator,  // the fixed bytes after it
        Branch,     // a combined grammar's first output token: calls or the JSON answer
        Text,       // free text before the first call
        Body,       // the JSON answer
        Header,     // `\n<function=NAME>\n` after a `<tool_call>` token
        Parameters, // a parameter opener or the call's close
        Name,
        ValueStart, // the `\n` after the name's `>`
        Value,
        AfterValue, // the `\n` after `</parameter>`
        CallClose,
        AfterCall,
    };

    struct Cursor {
        Phase phase           = Phase::Text;
        std::uint32_t matched = 0; // bytes of the current literal matched
        std::string name;
        MarkdownCodeTracker code; // the free text's Markdown code state (`quotes`)
        // The call's declared names when it may name only those (`declared`), else null.
        const std::vector<std::string>* declared = nullptr;
        bool after_value = false; // the parameter opener being read follows a raw value's terminator

        // The next phase of the same call: its declared names and whether a value came before stay.
        void enter(Phase next) {
            phase   = next;
            matched = 0;
            name.clear();
        }
    };

    // The current call's names: a simulation reads the committed ones and keeps its own additions.
    struct Names {
        const std::vector<std::string>* base;
        bool cleared = false;
        std::vector<std::string> added;

        [[nodiscard]] bool contains(const std::string& name) const {
            return std::find(added.begin(), added.end(), name) != added.end() ||
                   (!cleared && std::find(base->begin(), base->end(), name) != base->end());
        }
        void clear() {
            cleared = true;
            added.clear();
        }
        void apply(std::vector<std::string>& target) {
            if (cleared) { target.clear(); }
            for (auto& name : added) { target.push_back(std::move(name)); }
        }
    };

    struct Snapshot {
        Cursor cursor;
        std::vector<std::string> names;
    };

    static bool in_call(Phase phase) noexcept { return phase >= Phase::Header; }

    // Whether the `<tool_call>` token opens the next call after a call's closing lines.
    bool opens_call(const Cursor& cursor) const noexcept { return parallel_ && cursor.matched != 0; }

    bool quoted(const Cursor& cursor, std::int32_t token) const noexcept {
        return quotes_rule_ && token == tool_call_ && cursor.phase == Phase::Text && cursor.code.in_code();
    }

    bool step(Cursor& cursor, Names& names, std::int32_t token) const {
        if (quoted(cursor, token)) {
            cursor.code.feed(kToolCallOpen); // text, spelled as the decoder publishes it
            return true;
        }
        if (token == tool_call_) {
            // The token opens a call from free text, a combined grammar's first token, or after a
            // call's closing lines and their `\n`; a value holds it as its spelling.
            if (cursor.phase == Phase::AfterCall && !opens_call(cursor)) { return false; }
            if (cursor.phase == Phase::Text || cursor.phase == Phase::Branch || cursor.phase == Phase::AfterCall) {
                cursor = {.phase = Phase::Header};
                names.clear();
                return true;
            }
            return cursor.phase != Phase::Value || feed(cursor, names, kToolCallOpen);
        }
        if (cursor.phase == Phase::Reasoning) {
            if (token == *reasoning_close_) { cursor = {.phase = separator_ != 0 ? Phase::Separator : body_phase_}; }
            return true;
        }
        if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_->size()) { return true; }
        const std::string& bytes = (*vocabulary_)[static_cast<std::size_t>(token)];
        if (cursor.phase == Phase::Text) {
            if (quotes_rule_) { cursor.code.feed(bytes); }
            return true;
        }
        return feed(cursor, names, bytes);
    }

    // False when the bytes repeat a parameter name of the call.
    bool feed(Cursor& cursor, Names& names, std::string_view bytes) const {
        for (const char byte : bytes) {
            switch (cursor.phase) {
            case Phase::Reasoning:
            case Phase::Text:
            case Phase::Body: break;
            case Phase::Separator:
                if (++cursor.matched == separator_) { cursor = {.phase = body_phase_}; }
                break;
            case Phase::Branch: cursor.phase = Phase::Body; break;
            case Phase::Header:
                if (byte == '>') {
                    cursor.matched  = 1;
                    cursor.declared = declared_names(cursor.name);
                } else if (cursor.matched != 0) {
                    cursor.enter(Phase::Parameters);
                    cursor.after_value = false;
                } else {
                    cursor.name.push_back(byte);
                }
                break;
            case Phase::Parameters:
                if (byte == kParameterOpen[cursor.matched]) {
                    if (++cursor.matched == kParameterOpen.size()) { cursor.enter(Phase::Name); }
                } else {
                    // `</`: the call's close.
                    const std::uint32_t matched = cursor.matched + 1;
                    cursor.enter(Phase::CallClose);
                    cursor.matched = matched;
                }
                break;
            case Phase::Name:
                if (byte != '>') {
                    cursor.name.push_back(byte);
                    break;
                }
                if (cursor.declared != nullptr && cursor.after_value &&
                    (std::find(cursor.declared->begin(), cursor.declared->end(), cursor.name) ==
                         cursor.declared->end() ||
                     names.contains(cursor.name))) {
                    // A parameter line the call cannot take: the raw value's text, which goes on.
                    cursor.enter(Phase::Value);
                    break;
                }
                if (names_rule_ && names.contains(cursor.name)) { return false; }
                names.added.push_back(std::move(cursor.name));
                cursor.enter(Phase::ValueStart);
                break;
            case Phase::ValueStart: cursor.enter(Phase::Value); break;
            case Phase::Value: value_byte(cursor, byte); break;
            case Phase::AfterValue:
                cursor.enter(Phase::Parameters);
                cursor.after_value = true;
                break;
            case Phase::CallClose:
                if (++cursor.matched == kCallClose.size()) { cursor.enter(Phase::AfterCall); }
                break;
            case Phase::AfterCall:
                // The `\n` before the next call's token. Other text after the call's closers is the
                // last raw value's (the grammar reads those closers as value text then), and the call
                // goes on with its names.
                if (cursor.matched == 0 && byte == '\n') {
                    cursor.matched = 1;
                    break;
                }
                {
                    const bool newline = cursor.matched != 0;
                    cursor.enter(Phase::Value);
                    if (newline) { value_byte(cursor, '\n'); }
                    value_byte(cursor, byte);
                }
                break;
            }
        }
        return true;
    }

    // A raw value's byte; `\n` occurs in its terminator only at its start.
    static void value_byte(Cursor& cursor, char byte) {
        if (byte == kValueEnd[cursor.matched]) {
            if (++cursor.matched == kValueEnd.size()) { cursor.enter(Phase::AfterValue); }
        } else {
            cursor.matched = byte == kValueEnd[0] ? 1 : 0;
        }
    }

    // The declared names of the tool a call header `\n<function=NAME` names, when its call may name only
    // those; null otherwise.
    const std::vector<std::string>* declared_names(std::string_view header) const {
        constexpr std::string_view kFunctionOpen = "\n<function=";
        if (!declared_ || !header.starts_with(kFunctionOpen)) { return nullptr; }
        header.remove_prefix(kFunctionOpen.size());
        for (const auto& [tool, names] : *declared_) {
            if (tool == header) { return &names; }
        }
        return nullptr;
    }

    std::shared_ptr<const std::vector<std::string>> vocabulary_;
    std::shared_ptr<const std::vector<std::int32_t>> tag_ends_;
    std::int32_t tool_call_;
    std::optional<std::int32_t> reasoning_close_;
    std::size_t separator_;
    Phase body_phase_;
    bool parallel_;
    bool names_rule_;
    bool quotes_rule_;
    std::shared_ptr<const DeclaredNames> declared_;
    Cursor cursor_;
    std::vector<std::string> names_;
    Snapshot committed_;
};

// A non-strict tool's parameter list when its call names only its declared parameters
// (Contract::declared_names): each a raw string, once, in any order (the companion keeps them unique).
std::string declared_arguments_schema(const Contract::Tool& tool) {
    Json properties = Json::object();
    for (const auto& parameter : tool.parameters) { properties[parameter.name] = {{"type", "string"}}; }
    return Json{{"type", "object"}, {"properties", std::move(properties)}, {"additionalProperties", false}}.dump();
}

xgrammar::Grammar arguments(const Contract::Tool& tool, bool declared) {
    try {
        std::string source;
        if (tool.strict)
            source = text::prepare_json_schema(tool.grammar_schema_json.empty() ? tool.schema_json
                                                                               : tool.grammar_schema_json);
        else if (declared)
            source = declared_arguments_schema(tool);
        else
            source = R"({"type":"object","additionalProperties":{"type":"string"}})";
        // Declared names: any order, and a raw value also reads a quoted parameter line as its text.
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, declared, xgrammar::JSONFormat::kQwenXML, {}, declared);
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
                              ? RequestErrorKind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? RequestErrorKind::UnsatisfiableJsonSchema
                              : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer,
                           RequestErrorSource::Tools);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters",
                           RequestErrorSource::Tools);
    }
}

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body,
                        std::int32_t tool_call) {
    std::optional<xgrammar::Grammar> json;
    if (body) json = text::build_json_grammar(*body);
    if (json && contract.tools.empty()) return *json;
    xgrammar::GrammarBuilder builder;
    // Free text is token-level: any token but the `<tool_call>` token, which opens the calls.
    if (contract.tools.empty())
        return builder.Get(builder.AddRuleWithHint(
            "root", builder.AddTokenTagDispatch({{}, false, {tool_call}})));
    std::vector<int32_t> choices;
    std::optional<int32_t> basic_arguments;
    for (const auto& tool : contract.tools) {
        // Open non-strict calls share one parameter list; a strict or declared one is the tool's own.
        const bool declared = contract.declared_names(tool);
        const bool shared   = !tool.strict && !declared;
        const auto body     = shared && basic_arguments
                                  ? *basic_arguments
                                  : xgrammar::SubGrammarAdder::Apply(&builder, arguments(tool, declared));
        if (shared) basic_arguments = body;
        choices.push_back(builder.AddSequence(
            {builder.AddByteString("\n<function=" + tool.name + ">\n"), builder.AddRuleRef(body),
             builder.AddByteString("</function>\n</tool_call>")}));
    }
    const auto call_body = builder.AddRuleWithHint("call_body", builder.AddChoices(choices));
    const auto next_call =
        builder.AddSequence({builder.AddByteString("\n"), builder.AddTokenSet({tool_call}),
                             builder.AddRuleRef(call_body)});
    const auto tail  = contract.parallel ? builder.AddRepeatFromExpr("calls", next_call, 0, -1)
                                         : builder.AddEmptyStr();
    const auto first = builder.AddRuleWithHint(
        "calls", builder.AddSequence({builder.AddRuleRef(call_body), tail}));
    const auto root =
        (contract.required || body)
            ? builder.AddSequence({builder.AddTokenSet({tool_call}), builder.AddRuleRef(first)})
            : builder.AddTokenTagDispatch({{{tool_call, first}}, false, {}});
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
}
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::int32_t tool_call_token, const text::ReasoningFraming& reasoning,
                     std::span<const std::int32_t> continuation,
                     const std::optional<OutputConstraint>& body, text::GrammarCompiler::Compile mode) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel},
                  {"tool_call_token", tool_call_token}};
    if (body && !body->choices.empty())
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    if (body) identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    // The declared parameter names of each tool whose call may name only those (Contract::declared_names).
    auto declared = std::make_shared<std::vector<std::pair<std::string, std::vector<std::string>>>>();
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"] = Json::parse(tool.schema_json);
            // Failed compilations are cached too, including their declaration error location.
            entry["source_index"] = tool.declaration_index;
        } else if (contract.declared_names(tool)) {
            std::vector<std::string> names;
            for (const auto& parameter : tool.parameters) { names.push_back(parameter.name); }
            entry["declared"] = names;
            declared->emplace_back(tool.name, std::move(names));
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    // The grammar orders a strict tool's parameters, so only non-strict calls need the name rule; only
    // a grammar with free text before its calls (an auto choice without a JSON answer) needs quotes.
    // Every grammar with calls needs the rule for the token after a call's closing lines.
    std::unique_ptr<text::GrammarStateConstraint> companion;
    const bool names  = std::any_of(contract.tools.begin(), contract.tools.end(), [](const auto& tool) { return !tool.strict; });
    const bool quotes = !contract.tools.empty() && !contract.required && !body;
    if (!contract.tools.empty()) {
        companion = std::make_unique<ToolCallRules>(compiler.vocabulary(), compiler.tag_end_tokens(), tool_call_token,
                                                    reasoning, body.has_value() && !contract.required,
                                                    contract.parallel, names, quotes, std::move(declared));
    }
    // The reasoning may not write the `<tool_call>` token: a call there never runs, and the reasoning,
    // which cannot end the turn, would go on to its budget. Its spelling in ordinary tokens stays
    // reasoning text, as `</think>`'s does.
    text::ReasoningFraming framing = reasoning;
    framing.withheld_token         = tool_call_token;
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body, tool_call_token); }, framing,
        continuation, std::move(companion), mode);
}
} // namespace infernix::models::qwen3_5::frontend
