#include "models/qwen3_5/frontend/tool_grammar.h"
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

// The one rule of the constrained tool route the grammar cannot state: a call names each parameter
// once. Non-strict names are open (the grammar orders strict ones already), so a streamed call could
// otherwise repeat a key it has published. The companion reads only the canonical framing the grammar
// emits: `<tool_call>` (the token) `\n<function=NAME>\n`, then `<parameter=NAME>\nVALUE\n</parameter>\n`
// per parameter, then `</function>\n</tool_call>`, and `\n` before the next call's token. A value
// ends at its first `\n</parameter>` (a raw value cannot hold it, a JSON value cannot spell it).
class ParameterNameConstraint final : public text::GrammarStateConstraint {
public:
    ParameterNameConstraint(std::shared_ptr<const std::vector<std::string>> vocabulary,
                            std::shared_ptr<const std::vector<std::int32_t>> tag_ends,
                            std::int32_t tool_call, const text::ReasoningFraming& reasoning, bool combined)
        : vocabulary_(std::move(vocabulary)), tag_ends_(std::move(tag_ends)), tool_call_(tool_call),
          reasoning_close_(reasoning.close_token), separator_(reasoning.separator.size()),
          body_phase_(combined ? Phase::Branch : Phase::Text) {
        cursor_.phase = reasoning_close_ ? Phase::Reasoning : body_phase_;
        committed_    = {cursor_, names_};
    }

    void filter(std::span<std::uint32_t> mask) override {
        if (!in_call(cursor_.phase)) { return; }
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

    bool step(Cursor& cursor, Names& names, std::int32_t token) const {
        if (token == tool_call_) {
            // The token opens a call from free text, a combined grammar's first token, or after a call.
            if (cursor.phase == Phase::Text || cursor.phase == Phase::Branch || cursor.phase == Phase::AfterCall) {
                cursor = {.phase = Phase::Header};
                names.clear();
            }
            return true;
        }
        if (cursor.phase == Phase::Reasoning) {
            if (token == *reasoning_close_) { cursor = {.phase = separator_ != 0 ? Phase::Separator : body_phase_}; }
            return true;
        }
        if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_->size()) { return true; }
        return feed(cursor, names, (*vocabulary_)[static_cast<std::size_t>(token)]);
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
                    cursor.matched = 1;
                } else if (cursor.matched != 0) {
                    cursor = {.phase = Phase::Parameters};
                }
                break;
            case Phase::Parameters:
                if (byte == kParameterOpen[cursor.matched]) {
                    if (++cursor.matched == kParameterOpen.size()) { cursor = {.phase = Phase::Name}; }
                } else {
                    // `</`: the call's close.
                    cursor = {.phase = Phase::CallClose, .matched = cursor.matched + 1};
                }
                break;
            case Phase::Name:
                if (byte != '>') {
                    cursor.name.push_back(byte);
                    break;
                }
                if (names.contains(cursor.name)) { return false; }
                names.added.push_back(std::move(cursor.name));
                cursor = {.phase = Phase::ValueStart};
                break;
            case Phase::ValueStart: cursor = {.phase = Phase::Value}; break;
            case Phase::Value:
                // `\n` occurs in the terminator only at its start.
                if (byte == kValueEnd[cursor.matched]) {
                    if (++cursor.matched == kValueEnd.size()) { cursor = {.phase = Phase::AfterValue}; }
                } else {
                    cursor.matched = byte == kValueEnd[0] ? 1 : 0;
                }
                break;
            case Phase::AfterValue: cursor = {.phase = Phase::Parameters}; break;
            case Phase::CallClose:
                if (++cursor.matched == kCallClose.size()) { cursor = {.phase = Phase::AfterCall}; }
                break;
            case Phase::AfterCall: break; // the `\n` before the next call's token
            }
        }
        return true;
    }

    std::shared_ptr<const std::vector<std::string>> vocabulary_;
    std::shared_ptr<const std::vector<std::int32_t>> tag_ends_;
    std::int32_t tool_call_;
    std::optional<std::int32_t> reasoning_close_;
    std::size_t separator_;
    Phase body_phase_;
    Cursor cursor_;
    std::vector<std::string> names_;
    Snapshot committed_;
};

xgrammar::Grammar arguments(const Contract::Tool& tool) {
    try {
        std::string source;
        if (tool.strict)
            source = text::prepare_json_schema(tool.grammar_schema_json.empty() ? tool.schema_json
                                                                               : tool.grammar_schema_json);
        else
            source = R"({"type":"object","additionalProperties":{"type":"string"}})";
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, false, xgrammar::JSONFormat::kQwenXML, {});
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
        const auto body = !tool.strict && basic_arguments
                              ? *basic_arguments
                              : xgrammar::SubGrammarAdder::Apply(&builder, arguments(tool));
        if (!tool.strict) basic_arguments = body;
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
                     const std::optional<OutputConstraint>& body) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel},
                  {"tool_call_token", tool_call_token}};
    if (body && !body->choices.empty())
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    if (body) identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"] = Json::parse(tool.schema_json);
            // Failed compilations are cached too, including their declaration error location.
            entry["source_index"] = tool.declaration_index;
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    // The grammar orders a strict tool's parameters, so only non-strict calls need the companion.
    std::unique_ptr<text::GrammarStateConstraint> companion;
    if (std::any_of(contract.tools.begin(), contract.tools.end(), [](const auto& tool) { return !tool.strict; })) {
        companion = std::make_unique<ParameterNameConstraint>(compiler.vocabulary(), compiler.tag_end_tokens(),
                                                              tool_call_token, reasoning,
                                                              body.has_value() && !contract.required);
    }
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body, tool_call_token); }, reasoning,
        continuation, std::move(companion));
}
} // namespace infernix::models::qwen3_5::frontend
