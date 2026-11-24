#include "models/inkling2/frontend/constraints.h"

#include "text/json_schema.h"
#include "text/tokenizer.h"

#include <xgrammar/xgrammar.h>
#include "grammar_builder.h"
#include "grammar_functor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infernix::models::inkling2 {
namespace {

using Json = nlohmann::ordered_json;

// A marker's spelling: not UTF-8, so no byte-level rule matches it (constraints.h).
constexpr char kMarkerSpelling[] = "\xff";

[[noreturn]] void refuse_tools(const std::string& message, std::string pointer = {}) {
    throw RequestError(RequestErrorKind::InvalidToolConstraint, message, std::move(pointer), RequestErrorSource::Tools);
}

// A call's JSON: {"name": NAME, "args": ARGS}, ARGS the strict tool's schema (its root an object) or any object.
xgrammar::Grammar call_json(const DeclaredTool& tool) {
    const std::string pointer = "/" + std::to_string(tool.declaration_index) + "/parameters";
    try {
        Json args = Json{{"type", "object"}};
        if (tool.strict) {
            args = Json::parse(text::prepare_json_schema(tool.parameters));
            if (!args.is_object() || args.value("type", Json()) != Json("object")) {
                refuse_tools("tool '" + tool.name + "': a strict tool's parameters must be an object schema", pointer);
            }
        }
        Json properties     = Json::object();
        properties["name"]  = Json{{"const", tool.name}};
        properties["args"]  = std::move(args);
        Json call           = Json::object();
        call["type"]        = "object";
        call["properties"]  = std::move(properties);
        call["required"]    = Json::array({"name", "args"});
        call["additionalProperties"] = false;
        // Insignificant whitespace in runs of at most 32, as a JSON answer takes it (text/json_schema.cpp).
        return xgrammar::Grammar::FromJSONSchema(call.dump(), /*any_whitespace=*/true, std::nullopt, std::nullopt,
                                                 /*strict_mode=*/false, /*max_whitespace_cnt=*/32, false, false);
    } catch (const RequestError& error) {
        if (error.source() == RequestErrorSource::Tools) { throw; }
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(), pointer + error.pointer(),
                           RequestErrorSource::Tools);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema ? RequestErrorKind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? RequestErrorKind::UnsatisfiableJsonSchema
                              : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(), pointer + error.pointer,
                           RequestErrorSource::Tools);
    } catch (const xgrammar::LogFatalError& error) {
        refuse_tools("tool '" + tool.name + "': " + error.what(), pointer);
    } catch (const nlohmann::json::exception& error) {
        refuse_tools("tool '" + tool.name + "': " + error.what(), pointer);
    }
}

// One turn's grammar (constraints.h), built rule by rule.
class TurnBuilder {
public:
    explicit TurnBuilder(const OutputMarkers& markers) : m_(markers), reserved_(grammar_markers(markers)) {}

    // Any thinking blocks, then the answer block of `content`.
    xgrammar::Grammar answer_turn(const xgrammar::Grammar& content) {
        const std::int32_t answer = answer_block(content);
        return finish(b_.AddSequence({star("thinking_blocks", ref(thinking_block())), b_.AddRuleRef(answer)}));
    }

    // The tool policy's turn, with `content` the JSON answer an auto choice may give instead of calls.
    xgrammar::Grammar tool_turn(const ToolPolicy& policy, const std::optional<xgrammar::Grammar>& content) {
        const std::int32_t thinking = thinking_block();
        if (policy.tools.empty()) {
            // No callable tool: thinking and text blocks only.
            const std::int32_t block = rule("block", b_.AddChoices({b_.AddSequence({ref(thinking)}),
                                                                    b_.AddSequence({ref(text_block())})}));
            return finish(b_.AddSequence({star("blocks", ref(block))}));
        }
        const std::int32_t calls = calls_rule(policy);
        if (policy.required) {
            return finish(b_.AddSequence({star("thinking_blocks", ref(thinking)), ref(calls)}));
        }
        if (content) {
            const std::int32_t answer = answer_block(*content);
            const std::int32_t either =
                rule("answer_or_calls", b_.AddChoices({b_.AddSequence({ref(answer)}), b_.AddSequence({ref(calls)})}));
            return finish(b_.AddSequence({star("thinking_blocks", ref(thinking)), ref(either)}));
        }
        const std::int32_t block =
            rule("block", b_.AddChoices({b_.AddSequence({ref(thinking)}), b_.AddSequence({ref(text_block())})}));
        const std::int32_t tail = rule("calls_tail", b_.AddChoices({b_.AddSequence({ref(calls)}), b_.AddEmptyStr()}));
        return finish(b_.AddSequence({star("blocks", ref(block)), ref(tail)}));
    }

private:
    std::int32_t token(TokenId id) { return b_.AddTokenSet({id}); }
    std::int32_t ref(std::int32_t rule_id) { return b_.AddRuleRef(rule_id); }
    std::int32_t rule(const std::string& name, std::int32_t body) { return b_.AddRuleWithHint(name, body); }
    std::int32_t star(const std::string& name, std::int32_t expr) { return b_.AddRepeatFromExpr(name, expr, 0, -1); }
    // Any token but a marker, repeated.
    std::int32_t free_text(const std::string& name) { return star(name, b_.AddExcludeTokenSet(reserved_)); }

    std::int32_t thinking_block() {
        return rule("thinking", b_.AddSequence({token(m_.message_model), token(m_.thinking), free_text("thinking_text"),
                                                token(m_.end_message)}));
    }

    std::int32_t text_block() {
        return rule("text", b_.AddSequence({token(m_.message_model),
                                            b_.AddTokenSet({m_.content[0], m_.content[1], m_.content[2]}),
                                            free_text("text_text"), token(m_.end_message)}));
    }

    std::int32_t answer_block(const xgrammar::Grammar& content) {
        const std::int32_t body = xgrammar::SubGrammarAdder::Apply(&b_, content);
        return rule("answer", b_.AddSequence({token(m_.message_model), token(m_.content[0]), ref(body),
                                              token(m_.end_message)}));
    }

    // One call block of any callable tool; then, when parallel, more.
    std::int32_t calls_rule(const ToolPolicy& policy) {
        std::vector<std::int32_t> alternatives;
        for (const DeclaredTool& tool : policy.tools) {
            const std::int32_t json = xgrammar::SubGrammarAdder::Apply(&b_, call_json(tool));
            const std::int32_t name =
                rule("call_name", b_.AddChoices({b_.AddSequence({b_.AddByteString(tool.name)}), b_.AddEmptyStr()}));
            alternatives.push_back(b_.AddSequence(
                {token(m_.message_model), ref(name), token(m_.tool_call), ref(json), token(m_.end_message)}));
        }
        const std::int32_t call = rule("call", b_.AddChoices(alternatives));
        const std::int32_t more = policy.parallel ? star("more_calls", ref(call)) : b_.AddEmptyStr();
        return rule("calls", b_.AddSequence({ref(call), more}));
    }

    xgrammar::Grammar finish(std::int32_t root) {
        return xgrammar::GrammarNormalizer::Apply(b_.Get(b_.AddRuleWithHint("root", root)));
    }

    const OutputMarkers& m_;
    std::vector<TokenId> reserved_;
    xgrammar::GrammarBuilder b_;
};

} // namespace

std::vector<TokenId> grammar_markers(const OutputMarkers& m) {
    return {m.message_model, m.thinking, m.content[0], m.content[1], m.content[2], m.tool_call, m.end_message};
}

std::unique_ptr<text::GrammarCompiler> make_grammar_compiler(const text::Tokenizer& tokenizer,
                                                           const OutputMarkers& markers, std::vector<TokenId> eos,
                                                           std::size_t cache_bytes) {
    std::vector<std::string> vocabulary(tokenizer.vocab_size());
    for (std::size_t id = 0; id < vocabulary.size(); ++id) {
        if (!tokenizer.is_valid_token(static_cast<TokenId>(id))) { continue; }
        const auto decoded = tokenizer.decoded_token(static_cast<TokenId>(id));
        if (!decoded.special) { vocabulary[id] = decoded.bytes; }
    }
    std::vector<TokenId> reserved = grammar_markers(markers);
    for (const TokenId id : reserved) {
        if (id < 0 || static_cast<std::size_t>(id) >= vocabulary.size()) {
            throw std::invalid_argument("inkling2 grammar: a marker is outside the vocabulary");
        }
        vocabulary[static_cast<std::size_t>(id)] = kMarkerSpelling;
    }
    return std::make_unique<text::GrammarCompiler>(std::move(vocabulary), std::move(eos), cache_bytes,
                                                   std::move(reserved));
}

std::optional<ToolPolicy> select_tool_policy(const ToolNames* declared, const ToolChoice& choice) {
    const std::vector<DeclaredTool> none;
    const std::vector<DeclaredTool>& tools = declared != nullptr ? declared->tools : none;
    if (choice.allowed_names) {
        for (const std::string& name : *choice.allowed_names) {
            if (std::none_of(tools.begin(), tools.end(), [&](const DeclaredTool& t) { return t.name == name; })) {
                refuse_tools("tool choice refers to undeclared tool: " + name);
            }
        }
    }
    const bool strict = std::any_of(tools.begin(), tools.end(), [](const DeclaredTool& t) { return t.strict; });
    if (choice.mode == ToolChoiceMode::Auto && choice.parallel && !choice.allowed_names &&
        choice.constraints == ToolConstraintMode::Automatic && !strict) {
        return std::nullopt;
    }
    ToolPolicy policy;
    policy.required = choice.mode == ToolChoiceMode::Required;
    policy.parallel = choice.parallel;
    if (choice.mode != ToolChoiceMode::None) {
        for (const DeclaredTool& tool : tools) {
            if (!choice.allowed_names || std::find(choice.allowed_names->begin(), choice.allowed_names->end(),
                                                   tool.name) != choice.allowed_names->end()) {
                policy.tools.push_back(tool);
            }
        }
    }
    if (policy.tools.empty()) {
        if (policy.required) { refuse_tools("required tool choice has no callable tools"); }
        if (tools.empty()) { return std::nullopt; } // nothing declared: the free route has no call to protect
    }
    return policy;
}

std::unique_ptr<text::GrammarSession> compile_turn_grammar(text::GrammarCompiler& compiler,
                                                           const OutputMarkers& markers,
                                                           const std::optional<OutputConstraint>& content,
                                                           const ToolPolicy* tools,
                                                           text::GrammarCompiler::Compile mode) {
    const bool calls = tools != nullptr && !tools->tools.empty();
    if (!calls && content) {
        // No call to make (no tool policy, or one that may call none): the content constraint alone.
        const text::ContentFraming framing{
            .key  = "inkling2",
            .wrap = [&](const xgrammar::Grammar& grammar) { return TurnBuilder(markers).answer_turn(grammar); }};
        return compiler.compile_framed(*content, framing, mode);
    }
    if (tools == nullptr) { throw std::logic_error("inkling2 grammar: nothing to constrain"); }
    if (calls && content && content->kind != OutputConstraintKind::JsonObject &&
        content->kind != OutputConstraintKind::JsonSchema) {
        throw RequestError(text::constraint_error_kind(content->kind),
                           "active tools can be combined with JSON output constraints");
    }
    Json identity{{"inkling2_tools", Json::array()},
                  {"required", tools->required},
                  {"parallel", tools->parallel},
                  {"markers", grammar_markers(markers)}};
    if (content) { identity["content"] = {{"kind", static_cast<int>(content->kind)}, {"source", content->source}}; }
    for (const DeclaredTool& tool : tools->tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["parameters"]   = tool.parameters;
            entry["source_index"] = tool.declaration_index; // failed compiles are cached with their error's place
        }
        identity["inkling2_tools"].push_back(std::move(entry));
    }
    const auto build = [&] {
        std::optional<xgrammar::Grammar> answer;
        if (calls && content && !tools->required) { answer = text::build_json_grammar(*content); }
        try {
            return TurnBuilder(markers).tool_turn(*tools, answer);
        } catch (const xgrammar::LogFatalError& error) {
            refuse_tools(std::string("inkling2 tool grammar: ") + error.what());
        }
    };
    return compiler.compile_model(identity.dump(), build, text::ReasoningFraming{}, {}, nullptr, mode);
}

} // namespace infernix::models::inkling2
