#include "models/qwen3_5/frontend/tool_contract.h"
#include "text/json_schema.h"
#include "text/json_input.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace infernix::models::qwen3_5::frontend {
namespace {
using Json                = nlohmann::ordered_json;
using Contract            = ToolCallOutputContract;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;
using NormalizationPolicy = Contract::NormalizationPolicy;

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

[[noreturn]] void fail(std::string message, std::string path = {}) {
    throw RequestError(RequestErrorKind::InvalidToolConstraint, std::move(message),
                       std::move(path));
}

std::string parameter_path(const Contract::Tool& tool) {
    return "/" + std::to_string(tool.declaration_index) + "/parameters";
}

bool valid_parameter_name(std::string_view name) {
    return !name.empty() && name.find_first_of("<>\r\n") == std::string_view::npos;
}

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

const Json& resolve(const Json& node, const Json& document, std::set<std::string>& seen) {
    if (!node.is_object()) return node;
    if (node.contains("$ref") && node["$ref"].is_string()) {
        const auto ref = node["$ref"].get<std::string>();
        if ((ref != "#" && !ref.starts_with("#/")) || !seen.insert(ref).second)
            fail("tool schema requires resolvable local references");
        return resolve(document.at(Json::json_pointer(ref.substr(1))), document, seen);
    }
    if (node.contains("allOf") && node["allOf"].is_array() && node["allOf"].size() == 1)
        return resolve(node["allOf"][0], document, seen);
    return node;
}

std::uint8_t value_type(const Json& value) {
    if (value.is_string()) return type_bit(SchemaType::String);
    if (value.is_boolean()) return type_bit(SchemaType::Boolean);
    if (value.is_null()) return type_bit(SchemaType::Null);
    if (value.is_object()) return type_bit(SchemaType::Object);
    if (value.is_array()) return type_bit(SchemaType::Array);
    // JSON Schema classifies numbers by value: 1.0 belongs to the integer domain too.
    const bool integer =
        value.is_number_integer() ||
        (value.is_number_float() && std::floor(value.get<double>()) == value.get<double>());
    return type_bit(integer ? SchemaType::Integer : SchemaType::Number);
}

constexpr std::uint8_t kAnyType = 0x7f;

// The JSON types a schema admits, read from its type-bearing keywords alone: `type`, `const`,
// `enum`, a local `$ref`, `allOf` (each branch narrows them) and `anyOf` / `oneOf` (the union of
// their branches). It validates nothing: a keyword it cannot read (an invalid type, a remote or
// missing reference, a reference cycle) narrows nothing, and `known` stays false when nothing
// narrowed the types.
struct TypeDomain {
    std::uint8_t bits = kAnyType;
    bool known        = false;

    void narrow(TypeDomain other) {
        bits &= other.bits;
        known = known || other.known;
    }
};

// `open` holds the schemas being read, so a reference cycle reads as unknown.
TypeDomain type_domain(const Json& schema, const Json& document, std::set<const Json*>& open) {
    if (schema.is_boolean()) return schema.get<bool>() ? TypeDomain{} : TypeDomain{0, true};
    if (!schema.is_object() || !open.insert(&schema).second) return {};
    TypeDomain result;
    if (const auto ref = schema.find("$ref"); ref != schema.end() && ref->is_string()) {
        const auto& pointer = ref->get_ref<const std::string&>();
        const Json* target  = nullptr;
        if (pointer == "#" || pointer.starts_with("#/")) {
            try {
                target = &document.at(Json::json_pointer(pointer.substr(1)));
            } catch (const Json::exception&) {
                // A missing target narrows nothing.
            }
        }
        if (target != nullptr) result.narrow(type_domain(*target, document, open));
    }
    if (const auto type = schema.find("type"); type != schema.end()) {
        TypeSet direct;
        if (compile_direct_types(*type, direct)) {
            // A number type admits the integers too.
            if (direct.bits & type_bit(SchemaType::Number))
                direct.bits |= type_bit(SchemaType::Integer);
            result.narrow({direct.bits, true});
        }
    }
    if (const auto value = schema.find("const"); value != schema.end())
        result.narrow({value_type(*value), true});
    if (const auto values = schema.find("enum"); values != schema.end() && values->is_array()) {
        std::uint8_t bits = 0;
        for (const Json& value : *values) bits |= value_type(value);
        result.narrow({bits, true});
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        const auto branches = schema.find(key);
        if (branches == schema.end() || !branches->is_array() || branches->empty()) continue;
        // An unknown branch admits every type: the union is known only when every branch is.
        TypeDomain combined{0, true};
        for (const Json& branch : *branches) {
            const TypeDomain domain = type_domain(branch, document, open);
            combined.bits |= domain.bits;
            combined.known = combined.known && domain.known;
        }
        result.narrow(combined);
    }
    if (const auto branches = schema.find("allOf");
        branches != schema.end() && branches->is_array())
        for (const Json& branch : *branches) result.narrow(type_domain(branch, document, open));
    open.erase(&schema);
    return result;
}

TypeDomain type_domain(const Json& schema, const Json& document) {
    std::set<const Json*> open;
    return type_domain(schema, document, open);
}

// The raw-text branches of a nullable string, as the grammar takes them: the open string schemas,
// the members of the finite (enum / const) ones, and whether null is one of the values.
struct TextBranches {
    Json open     = Json::array();
    Json members  = Json::array();
    bool nullable = false;
};

// Adds a branch without structure: type "string" ("null" in a type list makes it nullable), type
// "null", or an enum / const of strings and null (prepare_json_schema has dropped the members its
// type refuses). False when the schema is none of these.
bool add_text_branch(const Json& node, TextBranches& text) {
    if (!node.is_object()) return false;
    for (const char* key : {"anyOf", "oneOf", "allOf", "properties", "items", "$ref"})
        if (node.contains(key)) return false;
    bool string = false;
    bool null   = false;
    if (const auto type = node.find("type"); type != node.end()) {
        const Json names = type->is_array() ? *type : Json::array({*type});
        if (names.empty()) return false;
        for (const Json& name : names) {
            if (name == "string") {
                string = true;
            } else if (name == "null") {
                null = true;
            } else {
                return false;
            }
        }
    }
    const auto constant = node.find("const");
    const auto values   = node.find("enum");
    if (constant == node.end() && values == node.end()) {
        if (string) {
            Json branch    = node;
            branch["type"] = "string";
            text.open.push_back(std::move(branch));
        } else if (!null) {
            return false;
        }
        text.nullable = text.nullable || null;
        return true;
    }
    // A finite branch's values are its members, whatever its type list names.
    const Json members = constant != node.end() ? Json::array({*constant}) : *values;
    if (!members.is_array()) return false;
    for (const Json& member : members) {
        if (member.is_null()) {
            text.nullable = true;
        } else if (!member.is_string()) {
            return false;
        } else if (std::find(text.members.begin(), text.members.end(), member) ==
                   text.members.end()) {
            text.members.push_back(member);
        }
    }
    return true;
}

// A nullable string parameter (BX-06: OpenAI strict mode writes every optional field as one): a
// type list of "string" and "null", an enum or const of strings and null (NU-01), or an anyOf /
// oneOf whose branches are such strings or null (at least one string). Its model text is raw, as a
// string's: the grammar takes the string branches, their members, and the text `null`, which reads
// back as JSON null. Empty when the schema is not one; `null_member` is set when a finite branch
// lists the string "null", which that text cannot tell apart from null.
std::optional<Json> nullable_text_branches(const Json& source, const Json& document,
                                           bool& null_member) {
    std::set<std::string> seen;
    const Json& node = resolve(source, document, seen);
    if (!node.is_object()) return std::nullopt;
    TextBranches text;
    const bool any_of = node.contains("anyOf");
    if (any_of || node.contains("oneOf")) {
        for (const char* key : {"type", "enum", "const", "properties", "items", "allOf"})
            if (node.contains(key)) return std::nullopt;
        if (any_of == node.contains("oneOf")) return std::nullopt;
        const Json& alternatives = node[any_of ? "anyOf" : "oneOf"];
        if (!alternatives.is_array() || alternatives.empty()) return std::nullopt;
        for (const Json& alternative : alternatives) {
            std::set<std::string> branch_seen;
            if (!add_text_branch(resolve(alternative, document, branch_seen), text))
                return std::nullopt;
        }
    } else if (!add_text_branch(node, text)) {
        return std::nullopt;
    }
    if (!text.nullable || (text.open.empty() && text.members.empty())) return std::nullopt;
    if (std::find(text.members.begin(), text.members.end(), Json("null")) != text.members.end()) {
        null_member = true;
        return std::nullopt;
    }
    // An open branch without a pattern whose length bounds admit four characters takes the text
    // `null` already; otherwise it is one more member.
    const bool admits_null =
        std::any_of(text.open.begin(), text.open.end(), [](const Json& branch) {
            return !branch.contains("pattern") && branch.value("minLength", 0) <= 4 &&
                   branch.value("maxLength", 4) >= 4;
        });
    if (!admits_null) text.members.push_back(Json("null"));
    Json branches = std::move(text.open);
    if (!text.members.empty())
        branches.push_back(Json{{"type", "string"}, {"enum", std::move(text.members)}});
    return branches.size() == 1 ? branches[0] : Json{{"anyOf", std::move(branches)}};
}

void prepare_strict_parameters(Contract::Tool& tool) {
    const Json document = Json::parse(text::prepare_json_schema(tool.schema_json));
    std::set<std::string> seen;
    const Json& root = resolve(document, document, seen);
    if (!root.is_object() || root.value("type", std::string("object")) != "object")
        fail("tool '" + tool.name + "' requires an object parameter schema", parameter_path(tool));
    if (!root.contains("type") || !root.contains("additionalProperties") ||
        root["additionalProperties"] != false)
        fail("strict tool '" + tool.name + "' requires type object and additionalProperties=false",
             parameter_path(tool));
    for (const char* assertion : {"const", "enum", "anyOf", "oneOf"})
        if (root.contains(assertion))
            fail("tool '" + tool.name + "' requires a single declared object parameter list",
                 parameter_path(tool) + "/" + assertion);
    tool.parameters.clear();
    tool.grammar_schema_json.clear();
    if (!root.contains("properties")) return;
    if (!root["properties"].is_object())
        fail("tool properties must be an object", parameter_path(tool));
    Json grammar_document = document;
    bool rewritten        = false;
    for (const auto& [name, schema] : root["properties"].items()) {
        if (!valid_parameter_name(name))
            fail("parameter name cannot be represented in Qwen tool format: " + name,
                 parameter_path(tool));
        Contract::Parameter parameter;
        parameter.name = name;
        // The grammar admits values of these types alone, so the decoder reads them as such.
        const auto types     = type_domain(schema, document).bits;
        parameter.policy     = NormalizationPolicy::DeclaredTypes;
        parameter.types.bits = types;
        const bool string    = (types & type_bit(SchemaType::String)) != 0;
        if (string && types == (type_bit(SchemaType::String) | type_bit(SchemaType::Null))) {
            // Raw text with null as one of its values: the grammar takes the string branches.
            bool null_member               = false;
            const std::optional<Json> text = nullable_text_branches(schema, document, null_member);
            if (null_member)
                fail("strict tool '" + tool.name + "' parameter '" + name +
                         "' lists the string \"null\" beside null, which its raw text cannot "
                         "tell apart",
                     parameter_path(tool));
            if (!text || &root != &document)
                fail("strict tool '" + tool.name + "' parameter '" + name +
                         "' mixes raw string and JSON value types beyond a nullable string",
                     parameter_path(tool));
            grammar_document["properties"][name] = *text;
            rewritten                            = true;
            parameter.nullable                   = true;
        } else if (string && types != type_bit(SchemaType::String)) {
            fail("strict tool '" + tool.name + "' parameter '" + name +
                     "' mixes raw string and JSON value types",
                 parameter_path(tool));
        }
        parameter.encoding = string ? Contract::Encoding::RawString : Contract::Encoding::Json;
        tool.parameters.push_back(std::move(parameter));
    }
    if (rewritten) { tool.grammar_schema_json = grammar_document.dump(); }
}
} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons,
                                std::span<const std::string> deferred_tool_jsons) {
    const std::size_t count = tool_jsons.size() + deferred_tool_jsons.size();
    if (count == 0) return {};
    auto result = std::make_shared<Contract>();
    std::set<std::string> names;
    for (std::size_t i = 0; i < count; ++i) {
        const std::string& source = i < tool_jsons.size() ? tool_jsons[i]
                                                          : deferred_tool_jsons[i - tool_jsons.size()];
        try {
            const auto parsed       = text::parse_json_numbers(source);
            const auto& declaration = parsed.value;
            const auto& function    = declaration.at("function");
            Contract::Tool tool;
            tool.name = function.at("name").get<std::string>();
            if (tool.name.empty() || !names.insert(tool.name).second)
                fail("tool names must be nonempty and unique", "/" + std::to_string(i));
            tool.strict = function.value("strict", false);
            if (tool.strict) {
                if (const auto pointer =
                        text::inexact_schema_number(parsed, "/function/parameters"))
                    throw RequestError(
                        RequestErrorKind::UnsupportedJsonSchema,
                        "numeric schema value cannot be preserved by the JSON number "
                        "representation",
                        "/" + std::to_string(i) + "/parameters" +
                            pointer->substr(std::string_view("/function/parameters").size()),
                        RequestErrorSource::Tools);
            }
            const Json schema = function.value(
                "parameters", Json{{"type", "object"}, {"properties", Json::object()}});
            tool.schema_json       = schema.dump();
            tool.declaration_index = i;
            // Non-strict output keeps the established best-effort normalization contract: the
            // types its schema names, through local references and enum / const values, guide
            // the value text.
            if (schema.is_object() && schema.contains("properties") &&
                schema["properties"].is_object()) {
                for (const auto& [name, property] : schema["properties"].items()) {
                    Contract::Parameter parameter;
                    parameter.name = name;
                    if (const TypeDomain types = type_domain(property, schema); types.known) {
                        parameter.policy     = NormalizationPolicy::DeclaredTypes;
                        parameter.types.bits = types.bits;
                    }
                    tool.parameters.push_back(std::move(parameter));
                }
            }
            result->tools.push_back(std::move(tool));
        } catch (const Json::exception& error) {
            fail(std::string("invalid tool declaration: ") + error.what(), "/" + std::to_string(i));
        }
    }
    return result;
}

std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice) {
    if (choice.allowed_names) {
        for (const auto& name : *choice.allowed_names) {
            if (!declarations ||
                std::none_of(declarations->tools.begin(), declarations->tools.end(),
                             [&](const auto& tool) { return tool.name == name; }))
                fail("tool choice refers to undeclared tool: " + name);
        }
    }
    if (declarations && choice.mode == ToolChoiceMode::Auto && choice.parallel &&
        !choice.allowed_names && choice.constraints == ToolConstraintMode::Automatic &&
        std::none_of(declarations->tools.begin(), declarations->tools.end(),
                     [](const auto& tool) { return tool.strict; }))
        return declarations;
    auto selected      = std::make_shared<Contract>();
    selected->required = choice.mode == ToolChoiceMode::Required;
    selected->parallel = choice.parallel;
    if (declarations && choice.mode != ToolChoiceMode::None) {
        for (const auto& tool : declarations->tools) {
            if (!choice.allowed_names ||
                std::find(choice.allowed_names->begin(), choice.allowed_names->end(), tool.name) !=
                    choice.allowed_names->end())
                selected->tools.push_back(tool);
        }
    }
    if (selected->tools.empty()) {
        if (selected->required) fail("required tool choice has no callable tools");
        if (!declarations || declarations->tools.empty()) return {};
        selected->constrained = true;
        return selected;
    }
    selected->constrained = selected->required || !selected->parallel ||
                            choice.allowed_names.has_value() ||
                            choice.constraints == ToolConstraintMode::Basic ||
                            std::any_of(selected->tools.begin(), selected->tools.end(),
                                        [](const auto& tool) { return tool.strict; });
    if (selected->constrained) {
        for (auto& tool : selected->tools) {
            if (!tool.strict) continue;
            try {
                prepare_strict_parameters(tool);
            } catch (const RequestError& error) {
                if (error.kind() == RequestErrorKind::InvalidToolConstraint ||
                    error.source() == RequestErrorSource::Tools)
                    throw;
                throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                                   parameter_path(tool) + error.pointer(),
                                   RequestErrorSource::Tools);
            } catch (const Json::exception& error) {
                fail("invalid parameters for tool '" + tool.name + "': " + error.what(),
                     parameter_path(tool));
            }
        }
    }
    return selected;
}
} // namespace infernix::models::qwen3_5::frontend
