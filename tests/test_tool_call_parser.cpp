#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_contract.h"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = infernix::models::qwen3_5::frontend;

// An empty contract: no declared tools, the parser's untyped legacy policy.
const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

// The model writes `<tool_call>` as its own token, so in these generated texts every spelling of the
// opener stands for that token; tests of the spelling from ordinary tokens pass their own offsets.
std::vector<std::size_t> tool_call_tokens(std::string_view text) {
    std::vector<std::size_t> found;
    for (std::size_t at = text.find(fi::kToolCallOpen); at != std::string_view::npos;
         at = text.find(fi::kToolCallOpen, at + 1)) {
        found.push_back(at);
    }
    return found;
}

fi::ParsedToolCallOutput parse(const std::string& text, std::size_t max_tool_name_length,
                              const fi::ToolCallOutputContract& contract, bool tolerant = false) {
    const std::vector<std::size_t> tokens = tool_call_tokens(text);
    return fi::parse_qwen_tool_call_output({text, tokens}, max_tool_name_length, contract,
                                           tolerant);
}

// Feeds bytes [begin, begin + size) of a generated text, with the `<tool_call>` tokens that start
// in them; a token may continue in the next piece.
std::string feed(fi::ToolCallOutputDecoder& decoder, std::string_view text, std::size_t begin = 0,
                 std::size_t size = std::string_view::npos) {
    const std::string_view piece = text.substr(begin, size);
    std::vector<std::size_t> tokens;
    for (const std::size_t at : tool_call_tokens(text)) {
        if (at >= begin && at - begin < piece.size()) { tokens.push_back(at - begin); }
    }
    return decoder.feed({piece, tokens});
}

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(definitions);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   infernix::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = parse(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = parse(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == infernix::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_untyped_parameters() {
    const auto contract = contract_for("get_weather", Json::object());
    const auto parsed   = parse("Calling weather.\n"
                                                            "<tool_call>\n"
                                                            "<function=get_weather>\n"
                                                            "<parameter=city>\nParis\n</parameter>\n"
                                                            "<parameter=days>\n2\n</parameter>\n"
                                                            "</function>\n"
                                                            "</tool_call>",
                                                          64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "untyped call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "untyped call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "untyped string inference changed");
    failures += check(args.at("days") == 2, "untyped JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const auto contract = contract_from_definitions(
        {tool_definition("first", Json::object()), tool_definition("second", Json::object())});
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = parse(text, 64, *contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        parse("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = parse(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = parse(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int test_parameter_delimiters_in_values() {
    using Reason        = infernix::ToolCallParseFallbackReason;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                    {"timeout", Json{{"type", "integer"}}}});
    int failures        = 0;

    // A closer the grammar cannot continue from is text the value quotes: the call is kept whole,
    // whether the quote is followed by a sibling parameter or ends the call.
    for (const std::string command :
         {std::string("echo '</parameter>'"), std::string("echo '</parameter>' && ls\n</param>")}) {
        for (const bool sibling : {false, true}) {
            const std::string text =
                sibling ? tool_call("bash", {{"command", command}, {"timeout", "30"}})
                        : tool_call("bash", {{"command", command}});
            const auto parsed = parse(text, 64, contract);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              "a value quoting a parameter closer lost its tool call");
            if (parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == command,
                              "a quoted parameter closer was not kept in the value");
            failures += check(!sibling || args.at("timeout") == 30,
                              "the parameter after a quoted closer was not parsed");
        }
    }

    // Markup the grammar could continue from stays ambiguous, so those calls still fall back.
    failures += check_rejected(tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}}),
                               contract, Reason::MalformedStructure,
                               "unbalanced nested parameter open was silently repaired");
    failures += check_rejected(tool_call("bash", {{"command", "echo '</parameter></function>'"}}),
                               contract, Reason::MalformedStructure,
                               "a closer followed by a function closer was guessed to be text");

    // Tolerant truncation: a closer at the region end still closes the value, and a quoted closer
    // inside a value the budget cut stays part of the kept partial value.
    const std::string open  = std::string("<") + "parameter=command>\n";
    const std::string close = std::string("</") + "parameter>";
    const std::string head  = "<tool_call>\n<function=bash>\n" + open;
    const auto closed_cut =
        parse(head + "ls\n" + close + "\n", 64, contract, true);
    const auto quoted_cut = parse(
        head + "echo '" + close + "' more", 64, contract, true);
    for (const auto* cut : {&closed_cut, &quoted_cut}) {
        failures += check(cut->is_tool_call_response && cut->tool_calls.size() == 1 &&
                              cut->diagnostics.fallback_reason == Reason::TruncatedTail,
                          "a tolerant cut call was not kept as a truncated tail");
    }
    if (closed_cut.tool_calls.size() == 1 && quoted_cut.tool_calls.size() == 1) {
        const Json closed_args = Json::parse(closed_cut.tool_calls.front().arguments_json);
        const Json quoted_args = Json::parse(quoted_cut.tool_calls.front().arguments_json);
        failures += check(closed_args.at("command") == "ls" &&
                              quoted_args.at("command") == "echo '" + close + "' more",
                          "a tolerant cut call ended its value at the wrong closer");
    }
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = parse(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        parse("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = parse(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = parse(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}},
             {"maybe_text", Json{{"type", Json::array({"string", "null"})}}},
             {"maybe_name",
              Json{{"anyOf", Json::array({Json{{"type", "string"}}, Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"},
                                                     {"maybe_text", "null"},
                                                     {"maybe_name", "Ann"}});
    const auto parsed      = parse(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("maybe_text").is_null() && args.at("maybe_name") == "Ann",
                          "a nullable string's null or text was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = parse(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == infernix::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = feed(decoder, text, 0, split);
        visible += feed(decoder, text, split);
        auto terminal = decoder.finish();
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (std::size_t at = 0; at < text.size(); ++at) { bytewise_visible += feed(bytewise, text, at, 1); }
    auto bytewise_terminal = bytewise.finish();
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = parse(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = parse(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_strict_structure_and_active_tool_set() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    failures +=
        check_rejected(malformed, contract, infernix::ToolCallParseFallbackReason::MalformedStructure,
                       "missing structural tags were accepted");

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    failures +=
        check_rejected(suffix, contract, infernix::ToolCallParseFallbackReason::TrailingContent,
                       "non-whitespace suffix was accepted");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    failures += check_rejected(missing_parameter_close, contract,
                               infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "missing parameter close was repaired");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    failures +=
        check_rejected(unknown_tool, contract, infernix::ToolCallParseFallbackReason::UndeclaredTool,
                       "undeclared tool name was accepted");

    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures +=
        check_rejected(invalid_name, contract, infernix::ToolCallParseFallbackReason::InvalidToolName,
                       "invalid function-name character was accepted");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const auto name_contract        = contract_for(name, Json::object());
    const std::string text          = tool_call(name);
    const auto anthropic            = parse(text, 128, name_contract);
    const auto openai               = parse(text, 64, name_contract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = parse(too_long_text, 128, name_contract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = parse(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_duplicate_tool_contracts_rejected() {
    const auto definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    try {
        (void)contract_from_definitions({definition, definition});
    } catch (const infernix::RequestError& error) {
        return check(error.kind() == infernix::RequestErrorKind::InvalidToolConstraint,
                     "duplicate declaration error kind");
    }
    return check(false, "duplicate declarations were accepted");
}

int test_all_or_nothing_structural_commit() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    return check_rejected(text, contract, infernix::ToolCallParseFallbackReason::MalformedStructure,
                          "partially valid tool-call region was partially committed");
}

int test_quoted_marker_before_real_call() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});
    const auto parsed = parse(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "bash",
                      "a quoted marker before the real call demoted the structured turn");
    failures += check(parsed.content == "explaining " + quoted + " then the real turn",
                      "quoted marker or intervening prose was not retained as content");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("command") == "echo ok", "recovered call arguments changed");
    }
    return failures;
}

int test_later_candidate_must_consume_the_end() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<parameter=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text =
        quoted + "\n" + tool_call("bash", {{"command", "echo ok"}}) + "\nstill explaining";
    const auto parsed = parse(text, 64, contract);

    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                          parsed.content == text && parsed.diagnostics.marker_seen &&
                          parsed.diagnostics.fallback_reason ==
                              infernix::ToolCallParseFallbackReason::MalformedStructure,
                      "a quoted marker before a non-terminal call was partially committed");
    return failures;
}

int test_incremental_quoted_marker_preserves_bytes() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 5;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += feed(decoder, text, offset, kChunk);
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "bash",
                      "incremental quoted marker hid the real tool call");
    failures += check(visible + terminal.content == "explaining " + quoted + " then the real turn",
                      "incremental quoted marker lost or duplicated bytes");
    failures +=
        check(terminal.diagnostics.marker_seen && terminal.diagnostics.structured_call_count == 1 &&
                  terminal.diagnostics.fallback_reason == infernix::ToolCallParseFallbackReason::None,
              "incremental quoted marker changed terminal diagnostics");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(output_contract_for("get_weather", Json::object()), 64);
    std::string visible;
    // A stop-string hold can split one token's bytes across published pieces.
    const std::string weather =
        "Calling weather.  \n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>";
    const std::size_t token_cut    = weather.find("call>");
    const std::size_t function_end = weather.find("\n</function>");
    visible += feed(legacy, weather, 0, token_cut);
    visible += feed(legacy, weather, token_cut, function_end - token_cut);
    visible += feed(legacy, weather, function_end);
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    const std::string enabled =
        "<tool_call>\n<function=configure>\n<parameter=enabled>\nTrue\n</parameter>\n</function>\n"
        "</tool_call>";
    const std::size_t value_cut = enabled.find("rue\n");
    boolean_visible += feed(boolean, enabled, 0, value_cut);
    boolean_visible += feed(boolean, enabled, value_cut, 1);
    boolean_visible += feed(boolean, enabled, value_cut + 1);
    auto boolean_terminal = boolean.finish();

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(output_contract_for("broken", Json::object()), 64);
    std::string restored;
    restored += feed(malformed, original, 0, 10);
    restored += feed(malformed, original, 10);
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += feed(ordinary, "ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += feed(partial, partial_original, 0, 6);
    partial_restored += feed(partial, partial_original, 6);
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              infernix::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += feed(decoder, text, offset, kChunk);
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

int test_free_tool_continuation() {
    const auto contract = output_contract_for("echo", Json{{"value", {{"type", "string"}}}});
    int failures        = 0;
    for (const auto& [prefix, suffix] :
         std::initializer_list<std::pair<std::string_view, std::string_view>>{
             {"Earlier <tool_ca", "broken"},
             {"Earlier <tool_ca", "ll>broken"},
             {"Earlier <tool_call>\n<function=echo>", "broken"}}) {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        decoder.initialize_continuation(prefix);
        std::string visible = feed(decoder, suffix);
        visible += decoder.finish(infernix::FinishReason::OutputLimit).content;
        failures += check(visible == suffix, "free continuation republished its prompt prefix");
    }
    const auto complete = tool_call("echo", {{"value", "hello"}});
    const auto split    = complete.find("hello") + 2;
    fi::ToolCallOutputDecoder decoder(contract, 64);
    decoder.initialize_continuation(std::string_view(complete).substr(0, split));
    (void)feed(decoder, complete, split);
    const auto result = decoder.finish();
    failures += check(result.tool_calls.size() == 1 && result.content.empty() &&
                          result.tool_calls[0].arguments_json == R"({"value":"hello"})",
                      "free continuation lost the completed call");
    bool rejected = false;
    try {
        fi::ToolCallOutputDecoder completed(contract, 64);
        completed.initialize_continuation(complete);
    } catch (const infernix::RequestError&) { rejected = true; }
    failures += check(rejected, "free continuation accepted an already published call");
    return failures;
}

// A byte that breaks a marker prefix may itself start the next marker, and a constrained
// contract recognizes only the <tool_call> form its grammar can emit.
int test_streamed_marker_boundaries() {
    int failures         = 0;
    const auto free_tool = output_contract_for("echo", Json{{"value", {{"type", "string"}}}});
    {
        fi::ToolCallOutputDecoder decoder(free_tool, 64);
        std::string visible = feed(decoder, "a<");
        visible += feed(decoder, tool_call("echo", {{"value", "hello"}}));
        const auto result = decoder.finish();
        visible += result.content;
        failures += check(visible == "a<" && result.tool_calls.size() == 1 &&
                              result.tool_calls[0].arguments_json == R"({"value":"hello"})",
                          "a broken marker prefix swallowed the marker that followed it");
    }
    const auto constrained = fi::select_tool_call_contract(free_tool, infernix::ToolChoice{});
    failures += check(constrained && constrained->constrained,
                      "default tool choice did not constrain tool calls");
    const std::string prose = "Quote <invoke name=\"echo\"> and <function_calls> as text.";
    fi::ToolCallOutputDecoder decoder(constrained, 64);
    std::string visible = feed(decoder, prose);
    const auto result   = decoder.finish();
    visible += result.content;
    failures += check(visible == prose && result.tool_calls.empty(),
                      "constrained content spelling an agent-harness marker left the content");
    return failures;
}

int test_claude_code_xml_markup_variants() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string standard_xml =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Initial setup\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_standard = parse(standard_xml, 128, contract);
    failures += check(parsed_standard.is_tool_call_response &&
                          parsed_standard.tool_calls.size() == 1 &&
                          parsed_standard.tool_calls.front().name == "TaskCreate",
                      "function name attribute syntax was not parsed");
    if (parsed_standard.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_standard.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Initial setup",
                          "function name attribute argument changed");
    }

    const std::string invoke_xml =
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Create tasks\n</parameter>\n</invoke>\n</tool_call>";
    const auto parsed_invoke = parse(invoke_xml, 128, contract);
    failures += check(parsed_invoke.is_tool_call_response && parsed_invoke.tool_calls.size() == 1 &&
                          parsed_invoke.tool_calls.front().name == "TaskCreate",
                      "invoke tag syntax was not parsed");

    const std::string function_calls_xml =
        "<function_calls>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Function calls container\n</parameter>\n</invoke>\n</function_calls>";
    const auto parsed_function_calls =
        parse(function_calls_xml, 128, contract);
    failures += check(parsed_function_calls.is_tool_call_response &&
                          parsed_function_calls.tool_calls.size() == 1 &&
                          parsed_function_calls.tool_calls.front().name == "TaskCreate",
                      "function_calls container syntax was not parsed");

    const std::string standalone_invoke =
        "Plan is ready:\n<invoke name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Standalone invoke\n</param>\n</invoke>";
    const auto parsed_standalone = parse(standalone_invoke, 128, contract);
    failures += check(parsed_standalone.is_tool_call_response &&
                          parsed_standalone.content == "Plan is ready:" &&
                          parsed_standalone.tool_calls.size() == 1 &&
                          parsed_standalone.tool_calls.front().name == "TaskCreate",
                      "standalone invoke after plan was not parsed");

    return failures;
}

int test_duplicate_parameters_keep_last_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    // A repeated identical parameter is the common agent-harness case: the second write leaves the
    // value alone, and the repair is still counted.
    const std::string identical_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nfirst\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_identical = parse(identical_dup, 64, contract);
    failures += check(parsed_identical.is_tool_call_response &&
                          parsed_identical.tool_calls.size() == 1,
                      "duplicate identical parameter was not accepted");
    if (parsed_identical.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_identical.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "first",
                          "repeated identical parameter value changed");
    }
    failures += check(parsed_identical.diagnostics.duplicate_parameters_repaired == 1,
                      "identical duplicate parameter repair was not recorded");

    // A conflicting repeat keeps the last value, matching the JSON-object rule the `=<name>`
    // markup already follows.
    const std::string conflicting_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nsecond\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_conflicting = parse(conflicting_dup, 64, contract);
    failures += check(parsed_conflicting.is_tool_call_response &&
                          parsed_conflicting.tool_calls.size() == 1,
                      "conflicting duplicate parameter fell back to text");
    if (parsed_conflicting.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_conflicting.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "second",
                          "conflicting duplicate parameter did not keep the last value");
    }
    failures += check(parsed_conflicting.diagnostics.duplicate_parameters_repaired == 1,
                      "conflicting duplicate parameter repair was not recorded");

    return failures;
}

int test_attribute_token_boundary() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string text =
        "<tool_call>\n<function filename=\"x\" name=\"TaskCreate\">\n"
        "<parameter filename=\"ignored\" name=\"description\">\nCreate task\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto parsed = parse(text, 128, contract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "TaskCreate",
                      "attribute token boundary failed to extract correct name");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Create task",
                          "parameter attribute token boundary failed");
    }
    return failures;
}

int test_mismatched_closing_tags_rejected() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string fn_invoke_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</invoke>\n</tool_call>";
    failures += check_rejected(fn_invoke_mismatch, contract,
                               infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "function opening with invoke closing tag was accepted");

    const std::string invoke_fn_mismatch =
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>";
    failures += check_rejected(invoke_fn_mismatch, contract,
                               infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "invoke opening with function closing tag was accepted");

    const std::string param_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</param>\n</function>\n</tool_call>";
    failures += check_rejected(param_mismatch, contract,
                               infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "parameter opening with param closing tag was accepted");

    const std::string param_open_mismatch =
        "<tool_call>\n<function name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>";
    failures += check_rejected(param_open_mismatch, contract,
                               infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "param opening with parameter closing tag was accepted");

    return failures;
}

int test_claude_code_plan_and_task_create_exact_repro() {
    const std::string task_create_def = tool_definition(
        "TaskCreate",
        Json{{"description", Json{{"type", "string"}}},
             {"task_type", Json{{"type", "string"}}},
             {"priority", Json{{"type", "integer"}}}});
    const std::string task_update_def = tool_definition(
        "TaskUpdate",
        Json{{"taskId", Json{{"type", "string"}}}, {"status", Json{{"type", "string"}}}});
    const auto contract = contract_from_definitions({task_create_def, task_update_def});

    const std::string full_response =
        "I have analyzed the repository requirements. Here is the implementation plan:\n\n"
        "### Plan\n"
        "1. Inspect existing CUDA kernels in `src/ops/softmax_attention/`\n"
        "2. Add test coverage for long context attention splits\n"
        "3. Update frontend tool call decoder\n\n"
        "Let me create the first task in the tracking system now:\n\n"
        "<tool_call>\n"
        "<function name=\"TaskCreate\">\n"
        "<parameter name=\"description\">\n"
        "Implement split-KV page-safety and bounded loops\n"
        "</parameter>\n"
        "<parameter name=\"task_type\">\n"
        "feature\n"
        "</parameter>\n"
        "<parameter name=\"priority\">\n"
        "1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const auto parsed = parse(full_response, 128, *contract);
    int failures      = 0;
    failures += check(parsed.is_tool_call_response,
                      "Claude Code Plan + TaskCreate failed to parse as tool call");
    failures += check(parsed.tool_calls.size() == 1, "tool call count != 1");
    failures += check(parsed.content.starts_with("I have analyzed"), "plan content prefix lost");
    failures += check(parsed.content.ends_with("tracking system now:"), "plan content tail lost");

    if (parsed.tool_calls.size() == 1) {
        const auto& call = parsed.tool_calls.front();
        failures += check(call.name == "TaskCreate", "tool name != TaskCreate");
        const Json args = Json::parse(call.arguments_json);
        failures += check(args.at("description") == "Implement split-KV page-safety and bounded loops",
                          "TaskCreate description argument changed");
        failures += check(args.at("task_type") == "feature", "TaskCreate task_type argument changed");
        failures += check(args.at("priority") == 1, "TaskCreate priority argument changed");
    }

    return failures;
}

} // namespace

int test_duplicate_parameter_keeps_last_value() {
    int failures = 0;
    const fi::ToolCallOutputContract contract =
        contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    const auto parsed = parse(duplicate, 64, contract);

    failures += check(parsed.is_tool_call_response, "duplicate parameter still fell back to text");
    failures += check(parsed.content.empty(), "duplicate parameter left prose behind");
    failures += check(parsed.tool_calls.size() == 1, "duplicate parameter did not yield one call");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json == R"({"value":"second"})",
                          "duplicate parameter did not keep the last value");
    }
    failures += check(parsed.diagnostics.fallback_reason ==
                          infernix::ToolCallParseFallbackReason::None,
                      "duplicate parameter still reported a fallback reason");
    failures += check(parsed.diagnostics.duplicate_parameters_repaired == 1,
                      "duplicate parameter repair was not recorded in diagnostics");
    return failures;
}

// Model bytes need not be valid UTF-8: a value cut inside a sequence (or a stray byte) must encode
// as U+FFFD rather than throw out of the parser.
int test_invalid_utf8_parameter_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;
    for (const std::string value : {std::string("ab\xC3"), std::string("a\xFF" "b"),
                                    std::string("\xE2\x82")}) {
        try {
            const auto parsed =
                parse(tool_call("configure", {{"value", value}}), 64,
                                                contract, true);
            failures += check(parsed.tool_calls.size() == 1,
                              "a call with invalid UTF-8 in its value was not parsed");
            if (parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("value").get<std::string>().find("\xEF\xBF\xBD") !=
                                  std::string::npos,
                              "invalid UTF-8 was not replaced by U+FFFD");
        } catch (const std::exception&) {
            failures += check(false, "invalid UTF-8 in a tool-call value threw");
        }
    }
    return failures;
}

int test_tolerant_recovery() {
    using Reason = infernix::ToolCallParseFallbackReason;
    int failures = 0;
    const std::string suffixed = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const auto tolerant = parse(suffixed, 64, contract, true);
    failures += check(tolerant.is_tool_call_response, "tolerant suffix was not recovered");
    failures += check(tolerant.tool_calls.size() == 1 && tolerant.tool_calls.front().name == "configure",
                      "tolerant suffix lost the recovered call");
    failures += check(tolerant.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant suffix was not flagged as a truncated tail");
    failures += check(tolerant.diagnostics.tolerant_recovered,
                      "tolerant suffix recovery was not reported as tolerant_recovered");
    const auto clean = parse(tool_call("configure", {{"value", "x"}}), 64,
                                                       contract, true);
    failures += check(clean.is_tool_call_response && !clean.diagnostics.tolerant_recovered,
                      "a well-formed call in tolerant mode was reported as tolerant_recovered");
    const auto strict = parse(suffixed, 64, contract);
    failures += check(!strict.is_tool_call_response, "strict suffix was recovered instead of text");
    failures +=
        check(!strict.diagnostics.tolerant_recovered, "strict mode reported tolerant_recovered");
    failures += check(strict.tool_calls.empty(), "strict suffix retained the recovered call");
    failures += check(strict.diagnostics.fallback_reason == Reason::TrailingContent,
                      "strict suffix was not flagged as trailing content");

    const std::string two = tool_call("first", {{"value", "a"}}) + "\n" + "<tool_call>\n<function=broken>";
    const auto tolerant_two = parse(two, 64, kLegacyContract, true);
    failures += check(tolerant_two.is_tool_call_response, "tolerant multi-call was not recovered");
    failures += check(tolerant_two.tool_calls.size() == 1 && tolerant_two.tool_calls.front().name == "first",
                      "tolerant multi-call kept the malformed second call");
    failures += check(tolerant_two.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "tolerant multi-call was not flagged as a truncated tail");
    const auto strict_two = parse(two, 64, kLegacyContract);
    failures += check(!strict_two.is_tool_call_response, "strict multi-call kept the recovered call");

    const auto decoder_contract =
        output_contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    fi::ToolCallOutputDecoder decoder(decoder_contract, 64, /*tolerant*/ true);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < suffixed.size(); offset += kChunk) {
        visible += feed(decoder, suffixed, offset, kChunk);
    }
    auto terminal = decoder.finish();
    failures += check(visible.empty() && terminal.content.empty(),
                      "tolerant increment leaked recovered bytes to visible content");
    failures += check(terminal.tool_calls.size() == 1,
                      "tolerant increment did not commit the recovered call");
    failures += check(terminal.diagnostics.fallback_reason == Reason::TruncatedTail &&
                          terminal.diagnostics.tolerant_recovered,
                      "tolerant increment lost the truncated-tail diagnostic");
    return failures;
}

int test_tolerant_truncated_final_call() {
    using Reason = infernix::ToolCallParseFallbackReason;
    const auto contract =
        output_contract_for("delete_file", Json{{"filePath", Json{{"type", "string"}}}});
    const std::string open_tag = std::string("<") + "parameter=filePath>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string truncated_tool_close =
        "Now let me verify.\n"
        "<tool_call>\n"
        "<function=delete_file>\n"
        + open_tag + "/tmp/out.js\n" + close_tag + "\n" + "</function>";
    const std::string truncated_function_close = "Now let me verify.\n"
                                                 "<tool_call>\n"
                                                 "<function=delete_file>\n"
                                                 + open_tag + "/tmp/out.js\n" + close_tag;
    const std::vector<std::pair<const char*, std::string>> cases = {
        {"missing tool close", truncated_tool_close},
        {"missing function close", truncated_function_close}};
    int failures = 0;
    for (const auto& [label, text] : cases) {
        const auto parsed = parse(text, 64, *contract, /*tolerant*/ true);
        failures += check(parsed.is_tool_call_response,
                          std::string("tolerant did not recover ") + label);
        failures += check(parsed.tool_calls.size() == 1,
                          std::string("tolerant recovered wrong call count for ") + label);
        if (parsed.tool_calls.size() == 1) {
            failures += check(parsed.tool_calls.front().name == "delete_file",
                              std::string("tolerant lost call name for ") + label);
            const Json arguments = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(arguments == Json{{"filePath", "/tmp/out.js"}},
                              std::string("tolerant lost arguments for ") + label);
        }
        failures += check(parsed.diagnostics.fallback_reason == Reason::TruncatedTail,
                          std::string("tolerant did not flag ") + label + " as truncated tail");
        const auto strict = parse(text, 64, *contract);
        failures += check(!strict.is_tool_call_response,
                          std::string("strict recovered a truncated final call: ") + label);
        failures += check(strict.tool_calls.empty(),
                          std::string("strict retained a truncated final call: ") + label);
    }
    return failures;
}

int test_tolerant_missing_function_close_bracket() {
    using Reason = infernix::ToolCallParseFallbackReason;
    const auto contract = output_contract_for(
        "memory",
        Json{{"command", Json{{"type", "string"}}}, {"path", Json{{"type", "string"}}}});
    const std::string open_cmd = std::string("<") + "parameter=command>\n";
    const std::string open_path = std::string("<") + "parameter=path>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string text =
        "Let me check my memory file first.\n"
        "<tool_call>\n"
        "<function=memory\n"
        + open_cmd + "str_replace\n" + close_tag + "\n"
        + open_path + "/memories/repo/notes.md\n" + close_tag + "\n"
        "</function>\n"
        "</tool_call>";
    int failures = 0;
    const auto tolerant = parse(text, 64, *contract, /*tolerant*/ true);
    failures += check(tolerant.is_tool_call_response,
                      "tolerant mode did not recover a missing closing bracket after the function name");
    failures += check(tolerant.tool_calls.size() == 1, "tolerant mode recovered wrong call count");
    if (tolerant.tool_calls.size() == 1) {
        const auto& call = tolerant.tool_calls.front();
        failures += check(call.name == "memory", "recovered call lost the function name");
        const Json args = Json::parse(call.arguments_json);
        failures += check(args.at("command").get<std::string>() == "str_replace",
                          "recovered call lost command arg");
        failures += check(args.at("path").get<std::string>() == "/memories/repo/notes.md",
                          "recovered call lost path arg");
    }
    failures += check(tolerant.diagnostics.fallback_reason == Reason::None,
                      "tolerant recovery of a missing bracket reported a spurious fallback reason");
    failures += check(tolerant.diagnostics.tolerant_recovered,
                      "a repaired missing bracket was not reported as tolerant_recovered");
    const auto strict = parse(text, 64, *contract);
    failures += check(!strict.is_tool_call_response,
                      "strict mode recovered a call with a missing closing bracket after the function name");
    failures += check(strict.tool_calls.empty(),
                      "strict mode retained an invalid tool name call");
    return failures;
}

int test_tolerant_undeclared_and_value_cut() {
    using Reason = infernix::ToolCallParseFallbackReason;
    const auto contract =
        output_contract_for("delete_file", Json{{"filePath", Json{{"type", "string"}}}});
    int failures = 0;
    const std::string open_tag = std::string("<") + "parameter=filePath>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const std::string undeclared = "<tool_call>\n"
                                   "<function=not_a_declared_tool>\n"
                                   + open_tag + "/tmp/out.js\n" + close_tag + "\n"
                                   "</function>\n"
                                   "</tool_call>";
    const auto tolerant = parse(undeclared, 64, *contract, true);
    failures += check(tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
                          tolerant.tool_calls.front().name == "not_a_declared_tool",
                      "tolerant mode did not keep the undeclared-name call structured");
    failures += check(tolerant.diagnostics.fallback_reason == Reason::None,
                      "tolerant undeclared call reported a fallback reason");
    failures += check(tolerant.diagnostics.tolerant_recovered,
                      "a kept undeclared-name call was not reported as tolerant_recovered");
    const auto strict = parse(undeclared, 64, *contract);
    failures += check(!strict.is_tool_call_response &&
                          strict.diagnostics.fallback_reason == Reason::UndeclaredTool,
                      "strict mode did not reject the undeclared-name call");

    const std::string value_cut = "Now\n"
                                  "<tool_call>\n"
                                  "<function=delete_file>\n"
                                  + open_tag + "/tmp/out";
    const auto cut_tolerant = parse(value_cut, 64, *contract, true);
    failures += check(cut_tolerant.is_tool_call_response && cut_tolerant.tool_calls.size() == 1 &&
                          cut_tolerant.tool_calls.front().name == "delete_file",
                      "tolerant mode did not keep the value-cut call");
    if (cut_tolerant.tool_calls.size() == 1) {
        const Json args = Json::parse(cut_tolerant.tool_calls.front().arguments_json);
        failures += check(args.at("filePath").get<std::string>() == "/tmp/out",
                          "tolerant mode lost the partial value-cut argument");
    }
    failures += check(cut_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail &&
                          cut_tolerant.diagnostics.tolerant_recovered,
                      "tolerant value-cut was not flagged as a recovered truncated tail");
    const auto cut_strict = parse(value_cut, 64, *contract);
    failures += check(!cut_strict.is_tool_call_response &&
                          cut_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                      "strict mode did not reject the value-cut call");

    // A call cut after the name but before any parameter completed carries no arguments, so
    // even tolerant mode returns the region as text (with the reason recorded).
    const std::string name_only = "<tool_call>\n"
                                  "<function=delete_file>\n";
    const auto name_tolerant = parse(name_only, 64, *contract, true);
    failures += check(!name_tolerant.is_tool_call_response && name_tolerant.tool_calls.empty(),
                      "tolerant mode kept a zero-parameter truncated call");
    failures += check(name_tolerant.diagnostics.fallback_reason == Reason::TruncatedTail &&
                          !name_tolerant.diagnostics.tolerant_recovered,
                      "zero-parameter truncation was not a truncated tail returned as text");
    const auto name_strict = parse(name_only, 64, *contract);
    failures += check(!name_strict.is_tool_call_response &&
                          name_strict.diagnostics.fallback_reason == Reason::MalformedStructure,
                      "strict mode misclassified a zero-parameter truncated call");
    return failures;
}

// Tolerant mode reads drifted function openers inside <tool_call> as <function=NAME>: a dropped
// keyword (`<=NAME>`), a colon (`<function:NAME>`) or the bare name (`<NAME>`, closed by
// </function> or </NAME>) only when NAME is declared; `<function =NAME>` is ordinary markup.
// Undeclared names in the drifted forms, and every drifted form in strict mode, stay text.
int test_tolerant_opener_drift() {
    using Reason        = infernix::ToolCallParseFallbackReason;
    const auto contract = output_contract_for(
        "read_file", Json{{"filePath", Json{{"type", "string"}}}});
    const std::string open_tag  = std::string("<") + "parameter=filePath>\n";
    const std::string close_tag = std::string("</") + "parameter>";
    const auto region = [&](const std::string& opener, const std::string& closer) {
        return "I'll read it.\n<tool_call>\n" + opener + "\n" + open_tag + "/tmp/a.md\n" +
               close_tag + "\n" + closer + "\n</tool_call>";
    };
    int failures = 0;
    const std::vector<std::pair<std::string, std::string>> recovered = {
        {"<=read_file>", "</function>"},
        {"<function:read_file>", "</function>"},
        {"<invoke:read_file>", "</invoke>"},
        {"<read_file>", "</function>"},
        {"<read_file>", "</read_file>"},
        {"<<=read_file>", "</function>"},
        {"<|im_start|>read_file>", "</function>"},
    };
    for (const auto& [opener, closer] : recovered) {
        const std::string text = region(opener, closer);
        const auto parsed      = parse(text, 64, *contract, true);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "read_file" &&
                              parsed.content == "I'll read it.",
                          "tolerant mode did not read the drifted opener " + opener + " ... " + closer);
        if (parsed.tool_calls.size() == 1) {
            failures += check(Json::parse(parsed.tool_calls.front().arguments_json) ==
                                  Json{{"filePath", "/tmp/a.md"}},
                              "the drifted opener " + opener + " lost its arguments");
        }
        failures += check(parsed.diagnostics.fallback_reason == Reason::None &&
                              parsed.diagnostics.tolerant_recovered,
                          "the drifted opener " + opener + " was not reported as a tolerant repair");
        const auto strict = parse(text, 64, *contract);
        failures += check(!strict.is_tool_call_response && strict.content == text,
                          "strict mode read the drifted opener " + opener);
    }

    // A space before '=' is ordinary header whitespace, in both modes.
    const std::string spaced = region("<function =read_file>", "</function>");
    for (const bool tolerant : {false, true}) {
        const auto parsed = parse(spaced, 64, *contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "read_file" &&
                              !parsed.diagnostics.tolerant_recovered,
                          "<function =NAME> was not read as a plain call");
    }

    // The drifted forms carry no keyword, so an undeclared name is prose, not a call (an
    // undeclared name under a real keyword stays structured in tolerant mode).
    for (const std::string opener : {"<=write_file>", "<function:write_file>", "<write_file>"}) {
        const std::string text = region(opener, "</function>");
        const auto parsed      = parse(text, 64, *contract, true);
        failures += check(!parsed.is_tool_call_response && parsed.content == text &&
                              parsed.diagnostics.fallback_reason == Reason::MalformedStructure,
                          "tolerant mode read the undeclared drifted opener " + opener);
        const auto legacy = parse(text, 64, kLegacyContract, true);
        failures += check(!legacy.is_tool_call_response,
                          "a drifted opener became a call with no declared tools: " + opener);
    }
    // A declared name that starts with the keyword is a bare name, not the keyword.
    {
        const auto prefixed = output_contract_for(
            "function_lookup", Json{{"filePath", Json{{"type", "string"}}}});
        const auto parsed = parse(
            region("<function_lookup>", "</function_lookup>"), 64, *prefixed, true);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().name == "function_lookup",
                          "a bare declared name starting with 'function' was misread");
    }
    // The streamed decoder agrees with the whole parse.
    {
        const std::string text = region("<read_file>", "</read_file>");
        fi::ToolCallOutputDecoder decoder(contract, 64, /*tolerant*/ true);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 5) {
            visible += feed(decoder, text, offset, 5);
        }
        const auto terminal = decoder.finish();
        failures += check(visible + terminal.content == "I'll read it." &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == "read_file",
                          "the streamed decoder did not read a bare declared opener");
    }
    return failures;
}

// Tolerant mode reads a parameter opener that lost its keyword, `<=NAME>`, as <parameter=NAME> when
// the call's tool declares NAME (Strata PR #3960: a drifting quant under Claude Code wrote
// `loop guard\n</parameter>\n<=output_mode>\ncontent`, and the next parameter ended up inside the
// previous value). An undeclared NAME, and every damaged opener in strict mode, stay value text.
int test_damaged_parameter_opener() {
    const auto contract = output_contract_for(
        "grep", Json{{"pattern", Json{{"type", "string"}}},
                     {"output_mode", Json{{"type", "string"}}},
                     {"path", Json{{"type", "string"}}}});
    const std::string open  = std::string("<") + "parameter=";
    const std::string close = std::string("</") + "parameter>";
    const auto region = [&](const std::string& body) {
        return "Searching.\n<tool_call>\n<function=grep>\n" + body + "</function>\n</tool_call>";
    };
    int failures = 0;
    struct Case {
        std::string label;
        std::string body;
        Json recovered;
    };
    const std::vector<Case> cases = {
        {"after a closer", open + "pattern>\nloop guard\n" + close + "\n<=output_mode>\ncontent\n" + close + "\n",
         Json{{"pattern", "loop guard"}, {"output_mode", "content"}}},
        {"first parameter", "<=pattern>\nloop guard\n" + close + "\n" + open + "path>\nsrc\n" + close + "\n",
         Json{{"pattern", "loop guard"}, {"path", "src"}}},
        {"two in a row", open + "pattern>\nx\n" + close + "\n<=output_mode>\ncount\n" + close + "\n<=path>\nsrc\n" +
                             close + "\n",
         Json{{"pattern", "x"}, {"output_mode", "count"}, {"path", "src"}}},
    };
    for (const auto& c : cases) {
        const std::string text = region(c.body);
        const auto parsed      = parse(text, 64, *contract, true);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.content == "Searching." && parsed.diagnostics.tolerant_recovered,
                          "tolerant mode did not repair the damaged opener (" + c.label + ")");
        if (parsed.tool_calls.size() == 1) {
            failures += check(Json::parse(parsed.tool_calls.front().arguments_json) == c.recovered,
                              "the damaged opener (" + c.label + ") gave other arguments: " +
                                  parsed.tool_calls.front().arguments_json);
        }
        // Strict mode keeps the old reading: the damaged opener's text belongs to a value or ends the call.
        const auto strict = parse(text, 64, *contract);
        failures += check(!strict.diagnostics.tolerant_recovered &&
                              (!strict.is_tool_call_response || strict.tool_calls.size() != 1 ||
                               Json::parse(strict.tool_calls.front().arguments_json) != c.recovered),
                          "strict mode repaired the damaged opener (" + c.label + ")");
    }
    // The reported string in strict mode: the second parameter stays inside the first value.
    {
        const auto strict = parse(region(cases.front().body), 64, *contract);
        failures += check(strict.is_tool_call_response && strict.tool_calls.size() == 1 &&
                              Json::parse(strict.tool_calls.front().arguments_json) ==
                                  Json{{"pattern", "loop guard\n" + close + "\n<=output_mode>\ncontent"}},
                          "strict mode changed its reading of the damaged opener");
    }
    // A name the tool does not declare is value text in tolerant mode too.
    {
        const std::string body = open + "pattern>\nloop guard\n" + close + "\n<=color>\nred\n" + close + "\n";
        const auto parsed      = parse(region(body), 64, *contract, true);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              Json::parse(parsed.tool_calls.front().arguments_json) ==
                                  Json{{"pattern", "loop guard\n" + close + "\n<=color>\nred"}},
                          "tolerant mode read an undeclared damaged opener as a parameter");
    }
    // The streamed decoder agrees with the whole parse.
    {
        const std::string text = region(cases.front().body);
        fi::ToolCallOutputDecoder decoder(contract, 64, /*tolerant*/ true);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += 7) { visible += feed(decoder, text, offset, 7); }
        const auto terminal = decoder.finish();
        failures += check(visible + terminal.content == "Searching." && terminal.tool_calls.size() == 1 &&
                              Json::parse(terminal.tool_calls.front().arguments_json) == cases.front().recovered,
                          "the streamed decoder did not repair the damaged opener");
    }
    return failures;
}

// A call opener inside a Markdown code fence or inline code of the answer is a quoted example:
// it stays text, with no byte lost, while calls at the top level still fire. Only the text before
// an opener decides, so every streamed chunking agrees with the whole output.
int test_markdown_quoted_calls() {
    const std::vector<std::string> definitions = {
        tool_definition("write", Json{{"path", Json{{"type", "string"}}}})};
    const auto contract = contract_from_definitions(definitions);
    const auto call     = tool_call("write", {{"path", "/tmp/a.txt"}});
    // (calls, content) of one output.
    using Outcome  = std::pair<std::size_t, std::string>;
    const auto run = [&](const std::string& text, bool tolerant, std::size_t width) {
        fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
        std::string content;
        for (std::size_t at = 0; at < text.size(); at += width) {
            content += feed(decoder, text, at, width);
        }
        auto terminal = decoder.finish();
        return Outcome(terminal.tool_calls.size(), content + terminal.content);
    };
    int failures       = 0;
    const auto outcome = [&](const std::string& text, bool tolerant) {
        const auto whole = parse(text, 64, *contract, tolerant);
        Outcome reference(whole.tool_calls.size(), whole.content);
        const std::vector<std::size_t> widths{1, 2, 3, 5, 7, 64, text.size()};
        for (const std::size_t width : widths) {
            if (run(text, tolerant, width) != reference) {
                failures += fail("a streamed chunking of width " + std::to_string(width) +
                                 " disagrees with the whole output: " + text.substr(0, 40));
            }
        }
        return reference;
    };
    const std::vector<std::string> quoted{
        "Format:\n```xml\n" + call + "\n```\nDone.", "Format:\n~~~\n" + call + "\n~~~\nDone.",
        "Use `" + call + "` like this.", "Use `x`, then `\n" + call + "`", "```\n" + call,
        "Text\n\n```python\nx = 1\n```\n```\n" + call};
    const std::vector<std::string> real{call,
                                        "Let me do it.\n\n" + call,
                                        "Let me do it. " + call,
                                        "```\ncode\n```\n" + call,
                                        "`a` and `b` " + call,
                                        "~~~\nx\n~~~\n\n" + call};
    for (const bool tolerant : {false, true}) {
        for (const std::string& body : quoted) {
            const Outcome got = outcome(body, tolerant);
            failures += check(got.first == 0 && got.second == body,
                              "a call quoted in Markdown code became a call or lost bytes: " +
                                  body.substr(0, 30));
        }
        for (const std::string& body : real) {
            failures += check(outcome(body, tolerant).first == 1,
                              "a top-level call after closed Markdown code did not fire: " +
                                  body.substr(0, 30));
        }
        failures += check(outcome(call + "\n" + call, tolerant).first == 2,
                          "two top-level calls did not both fire");
        // A quoted example before the real call stays text; the call still fires.
        const std::string mixed = "Example:\n```\n" + call + "\n```\nNow for real.\n\n" + call;
        const Outcome both      = outcome(mixed, tolerant);
        failures += check(both.first == 1 && both.second.starts_with("Example:\n```\n<tool_call>"),
                          "a quoted example before a real call was not kept as text");
    }
    return failures;
}

// A turn that ends inside its thinking can strand the calls it meant to make there. Tolerant mode
// returns them when the rest of the thinking from some marker is nothing but complete calls.
int test_open_reasoning_recovery() {
    const std::vector<std::string> definitions = {
        tool_definition("delete_file", Json{{"filePath", Json{{"type", "string"}}}}),
        tool_definition("read_file", Json{{"filePath", Json{{"type", "string"}}}})};
    const auto contract  = contract_from_definitions(definitions);
    const auto remove    = tool_call("delete_file", {{"filePath", "/tmp/stale.cfg"}});
    const auto read      = tool_call("read_file", {{"filePath", "/tmp/notes.md"}});
    const auto planning  = std::string("The stale config should go, so I'll call delete_file.\n");
    const auto recovered = [&](std::string_view reasoning,
                               infernix::FinishReason reason = infernix::FinishReason::StopToken,
                               bool tolerant = true, std::string_view content = {}) {
        fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
        std::string visible = feed(decoder, content);
        auto terminal       = decoder.finish(reason, {reasoning, tool_call_tokens(reasoning)});
        terminal.content    = visible + terminal.content;
        return terminal;
    };

    int failures = 0;
    {
        const auto terminal = recovered(planning + remove);
        failures += check(terminal.tool_calls.size() == 1 && terminal.content.empty() &&
                              terminal.diagnostics.recovered_from_reasoning &&
                              terminal.diagnostics.marker_seen &&
                              terminal.diagnostics.structured_call_count == 1 &&
                              !terminal.diagnostics.tolerant_recovered,
                          "a call stranded in open thinking was not recovered on its own");
        if (terminal.tool_calls.size() == 1) {
            failures += check(terminal.tool_calls[0].name == "delete_file" &&
                                  terminal.tool_calls[0].arguments_json ==
                                      R"({"filePath":"/tmp/stale.cfg"})",
                              "the recovered stranded call lost its name or argument");
        }
    }
    {
        const auto terminal = recovered(planning + read + "\n" + remove + "\n");
        failures += check(terminal.tool_calls.size() == 2 && terminal.content.empty() &&
                              terminal.tool_calls[0].name == "read_file" &&
                              terminal.tool_calls[1].name == "delete_file",
                          "parallel calls stranded in open thinking were not all recovered");
    }
    {
        // A complete call quoted in planning prose is not part of the final call run.
        const auto terminal =
            recovered("Earlier I wrote " + read + " but that was wrong.\n" + planning + remove);
        failures += check(terminal.tool_calls.size() == 1 && terminal.content.empty() &&
                              terminal.tool_calls[0].name == "delete_file",
                          "a call quoted earlier in the thinking joined the stranded call");
    }

    // Thinking that only quotes a marker, keeps deliberating after a call, cuts a call off or
    // names an undeclared tool keeps everything as reasoning and publishes nothing.
    const std::vector<std::pair<std::string, std::string>> rejected = {
        {"I could emit <tool_call> here, but the user only asked a question.\n",
         "a quoted marker"},
        {planning + remove + "\nOn second thought, I should ask first.", "trailing deliberation"},
        {planning + "<tool_call>\n<function=delete_file>\n<parameter=filePath>\n/tmp/stale.cfg\n",
         "a cut call"},
        {planning + tool_call("rename_file", {{"filePath", "/tmp/a"}}), "an undeclared tool"},
    };
    for (const auto& [reasoning, label] : rejected) {
        const auto terminal = recovered(reasoning);
        failures += check(terminal.tool_calls.empty() && terminal.content.empty() &&
                              !terminal.diagnostics.recovered_from_reasoning,
                          "open thinking with " + label + " produced a call or leaked content");
    }

    // Only a turn the model ended itself, with tolerant mode, free tools and no content fed.
    for (const infernix::FinishReason reason :
         {infernix::FinishReason::OutputLimit, infernix::FinishReason::ContextCapacity,
          infernix::FinishReason::Cancelled, infernix::FinishReason::StopString}) {
        failures += check(recovered(planning + remove, reason).tool_calls.empty(),
                          "a call was recovered from thinking cut by a limit, stop string or "
                          "cancellation");
    }
    failures += check(recovered(planning + remove, infernix::FinishReason::StopToken, false)
                          .tool_calls.empty(),
                      "strict mode recovered a call from open thinking");
    {
        const auto terminal =
            recovered(planning + remove, infernix::FinishReason::StopToken, true, "Answer.");
        failures += check(terminal.tool_calls.empty() && terminal.content == "Answer.",
                          "open thinking overrode a turn that fed content");
    }
    {
        const auto constrained = fi::select_tool_call_contract(contract, infernix::ToolChoice{});
        fi::ToolCallOutputDecoder decoder(constrained, 64, true);
        const std::string reasoning = planning + remove;
        failures += check(decoder
                              .finish(infernix::FinishReason::StopToken,
                                      {reasoning, tool_call_tokens(reasoning)})
                              .tool_calls.empty(),
                          "constrained tool output recovered a call from open thinking");
    }
    return failures;
}

// `<tool_call>` opens a call only where the model wrote its token. The same spelling from ordinary
// text tokens is text in the whole parse, the streamed decoder (strict and tolerant) and open-thinking
// recovery, and the bare `<function=...>` it wraps does not turn it into a call either.
int test_opener_token_identity() {
    const auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string call = tool_call("bash", {{"command", "ls"}});
    int failures           = 0;

    for (const bool tolerant : {false, true}) {
        const std::string label = tolerant ? " (tolerant)" : " (strict)";
        const auto spelled = fi::parse_qwen_tool_call_output({call, {}}, 64, *contract, tolerant);
        failures += check(!spelled.is_tool_call_response && spelled.content == call &&
                              !spelled.diagnostics.marker_seen,
                          "a <tool_call> spelled with ordinary tokens opened a call" + label);

        // A spelled call quoted before the model's token: only the token's region is the call.
        const std::string text = "The format is " + call + "\nRunning it now." + call;
        const std::vector<std::size_t> real{text.size() - call.size()};
        const auto parsed = fi::parse_qwen_tool_call_output({text, real}, 64, *contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.content == "The format is " + call + "\nRunning it now." &&
                              parsed.diagnostics.fallback_reason ==
                                  infernix::ToolCallParseFallbackReason::None,
                          "the token after a spelled call was not the only call" + label);

        // Streamed one byte at a time, with the token's bytes split across pieces, the decoder
        // publishes the spelled call and the prose as they arrive and holds only the real call.
        fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
        std::string visible;
        for (std::size_t at = 0; at < text.size(); ++at) {
            const std::vector<std::size_t> starts =
                at == real.front() ? std::vector<std::size_t>{0} : std::vector<std::size_t>{};
            visible += decoder.feed({std::string_view(text).substr(at, 1), starts});
        }
        failures += check(visible == "The format is " + call + "\nRunning it now.",
                          "the decoder held or dropped a spelled call" + label);
        const auto terminal = decoder.finish();
        failures += check(terminal.content.empty() && terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().arguments_json == R"({"command":"ls"})",
                          "the decoder did not publish the token's call" + label);
    }

    // Calls stranded in open thinking are recovered only from the token.
    const std::string planning = "I will list the directory.\n";
    const std::string thinking = planning + call;
    const std::vector<std::size_t> opener{planning.size()};
    for (const bool marked : {false, true}) {
        fi::ToolCallOutputDecoder decoder(contract, 64, true);
        const auto terminal = decoder.finish(
            infernix::FinishReason::StopToken,
            {thinking, marked ? std::span<const std::size_t>(opener) : std::span<const std::size_t>{}});
        failures += check(terminal.tool_calls.size() == (marked ? 1U : 0U) &&
                              terminal.diagnostics.recovered_from_reasoning == marked,
                          marked ? "a call stranded in thinking was not recovered from its token"
                                 : "a spelled call was recovered from open thinking");
    }
    return failures;
}

// The constrained route's calls as their steps streamed them.
struct StreamedCalls {
    std::vector<infernix::GeneratedToolCall> calls;
    std::vector<bool> complete;
    bool ordered = true;

    void take(fi::ToolCallOutputDecoder& decoder) {
        for (infernix::ToolCallDelta& delta : decoder.take_call_deltas()) {
            if (delta.index == calls.size()) {
                ordered = ordered && !delta.name.empty();
                calls.push_back({.name = delta.name, .arguments_json = {}});
                complete.push_back(false);
            } else {
                ordered = ordered && delta.index + 1U == calls.size() && delta.name.empty() &&
                          !complete.back();
            }
            calls.back().arguments_json += delta.arguments;
            complete.back() = delta.complete;
        }
    }
};

// Constrained calls stream their arguments JSON as they are generated, in fragments that concatenate
// to the terminal calls, whether the output arrives byte by byte or whole.
int test_constrained_call_streaming() {
    int failures             = 0;
    const auto declarations  = contract_from_definitions(
        {tool_definition("edit", Json{{"path", {{"type", "string"}}},
                                      {"count", {{"type", "integer"}}},
                                      {"flags", Json::object()}})});
    const auto contract      = fi::select_tool_call_contract(declarations, infernix::ToolChoice{});
    // The second call's value holds a terminator prefix that breaks off, and a closer without its
    // line break; a value ending in `\r` keeps it (the canonical framing is one line break each side).
    const std::string output = "Editing.\n" +
                               tool_call("edit", {{"path", "a \"quoted\" \\ path\n\tline"},
                                                  {"count", " 3 "},
                                                  {"flags", "[1, 2]"}}) +
                               "\n" + tool_call("edit", {{"count", ""}, {"path", "x\n</paramX</parameter>y\r"}});
    const std::string first  = R"({"path":"a \"quoted\" \\ path\n\tline","count":3,"flags":[1, 2]})";
    const std::string second = R"({"path":"x\n</paramX</parameter>y\r"})";
    for (const bool bytewise : {false, true}) {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        StreamedCalls streamed;
        std::string visible;
        bool streamed_early = false;
        if (bytewise) {
            for (std::size_t at = 0; at < output.size(); ++at) {
                visible += feed(decoder, output, at, 1);
                streamed.take(decoder);
                // The string value is published before its parameter closes.
                streamed_early = streamed_early || (!streamed.calls.empty() &&
                                                    streamed.calls[0].arguments_json == R"({"path":"a \"quo)");
            }
        } else {
            visible = feed(decoder, output);
            streamed.take(decoder);
        }
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        streamed.take(decoder);
        const std::string mode = bytewise ? " (byte by byte)" : " (whole)";
        failures += check(visible == "Editing." && terminal.content.empty(), "constrained content changed" + mode);
        failures += check(terminal.tool_calls.size() == 2 && terminal.tool_calls[0].arguments_json == first &&
                              terminal.tool_calls[1].arguments_json == second &&
                              terminal.diagnostics.structured_call_count == 2 &&
                              terminal.diagnostics.empty_arguments_omitted == 1,
                          "constrained terminal calls changed" + mode);
        failures += check(streamed.ordered && streamed.calls.size() == 2 && streamed.complete[0] &&
                              streamed.complete[1] && streamed.calls[0].name == "edit" &&
                              streamed.calls[0].arguments_json == first && streamed.calls[1].arguments_json == second,
                          "streamed call steps do not concatenate to the terminal calls" + mode);
        if (bytewise) { failures += check(streamed_early, "a string value waited for its parameter to close"); }
    }

    // A call the output limit cuts keeps the steps it streamed and stays out of the terminal calls.
    {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        StreamedCalls streamed;
        const std::string whole = tool_call("edit", {{"path", "p"}}) + "\n" + tool_call("edit", {{"path", "partial value"}});
        // Cut after "partial va": "lue", then `\n</parameter>\n</function>\n</tool_call>`.
        (void)feed(decoder, whole.substr(0, whole.size() - 3 - 38));
        streamed.take(decoder);
        const auto terminal = decoder.finish(infernix::FinishReason::OutputLimit);
        failures += check(terminal.tool_calls.size() == 1 && streamed.calls.size() == 2 && streamed.complete[0] &&
                              !streamed.complete[1] && streamed.calls[1].arguments_json == R"({"path":"partial va)",
                          "a cut constrained call was published or lost its streamed steps");
    }

    // A strict call streams its raw string escaped and its JSON value verbatim, delimited lexically.
    {
        const auto strict_declarations = contract_from_definitions({Json{
            {"type", "function"},
            {"function",
             {{"name", "record"},
              {"strict", true},
              {"parameters",
               {{"type", "object"},
                {"properties",
                 {{"message", {{"type", "string"}}},
                  {"payload",
                   {{"type", "object"},
                    {"properties", {{"text", {{"type", "string"}}}, {"number", {{"type", "integer"}}}}},
                    {"required", {"text", "number"}},
                    {"additionalProperties", false}}}}},
                {"required", {"message", "payload"}},
                {"additionalProperties", false}}}}}}.dump()});
        const auto strict = fi::select_tool_call_contract(strict_declarations, infernix::ToolChoice{});
        fi::ToolCallOutputDecoder decoder(strict, 64);
        StreamedCalls streamed;
        const std::string call =
            tool_call("record", {{"message", " \xe4\xbd\xa0\xe5\xa5\xbd\n"}, {"payload", R"({"text":"</parameter>\n","number":3})"}});
        for (std::size_t at = 0; at < call.size(); ++at) {
            (void)feed(decoder, call, at, 1);
            streamed.take(decoder);
        }
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        const std::string expected = "{\"message\":\" \xe4\xbd\xa0\xe5\xa5\xbd\\n\",\"payload\":{\"text\":\"</parameter>\\n\",\"number\":3}}";
        failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].arguments_json == expected &&
                              streamed.calls.size() == 1 && streamed.calls[0].arguments_json == expected,
                          "a strict call's raw string or JSON value streamed differently");
    }

    // A call an assistant prefix began streams from its start, the prefix's bytes included.
    {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        decoder.initialize_continuation("<tool_call>\n<function=edit>\n<parameter=path>\nab");
        StreamedCalls streamed;
        (void)feed(decoder, "c\n</parameter>\n</function>\n</tool_call>");
        streamed.take(decoder);
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].arguments_json == R"({"path":"abc"})" &&
                              streamed.calls.size() == 1 && streamed.calls[0].arguments_json == R"({"path":"abc"})",
                          "a continued call did not stream from its start");
    }

    // The grammar's companion names each parameter once, so a repeat is a disagreement, not a repair.
    {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        (void)feed(decoder, tool_call("edit", {{"path", "a"}, {"path", "b"}}));
        bool refused = false;
        try {
            (void)decoder.finish(infernix::FinishReason::StopToken);
        } catch (const std::logic_error&) { refused = true; }
        failures += check(refused, "a constrained call repeated a parameter name");
    }
    return failures;
}

int main() {
    int failures = 0;
    failures += test_constrained_call_streaming();
    failures += test_opener_token_identity();
    failures += test_free_tool_continuation();
    failures += test_untyped_parameters();
    failures += test_invalid_utf8_parameter_value();
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_parameter_delimiters_in_values();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_strict_structure_and_active_tool_set();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_duplicate_tool_contracts_rejected();
    failures += test_all_or_nothing_structural_commit();
    failures += test_quoted_marker_before_real_call();
    failures += test_later_candidate_must_consume_the_end();
    failures += test_incremental_quoted_marker_preserves_bytes();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_streamed_marker_boundaries();
    failures += test_claude_code_xml_markup_variants();
    failures += test_duplicate_parameters_keep_last_value();
    failures += test_attribute_token_boundary();
    failures += test_mismatched_closing_tags_rejected();
    failures += test_claude_code_plan_and_task_create_exact_repro();
    failures += test_tolerant_recovery();
    failures += test_tolerant_truncated_final_call();
    failures += test_tolerant_missing_function_close_bracket();
    failures += test_tolerant_undeclared_and_value_cut();
    failures += test_tolerant_opener_drift();
    failures += test_damaged_parameter_opener();
    failures += test_open_reasoning_recovery();
    failures += test_markdown_quoted_calls();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
