#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_contract.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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

// `cut_short`: the output ended at its budget (as opposed to the model's own stop).
fi::ParsedToolCallOutput parse(const std::string& text, std::size_t max_tool_name_length,
                              const fi::ToolCallOutputContract& contract, bool tolerant = false,
                              bool cut_short = false) {
    const std::vector<std::size_t> tokens = tool_call_tokens(text);
    return fi::parse_qwen_tool_call_output({text, tokens}, max_tool_name_length, contract,
                                           tolerant, {}, cut_short);
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

// Several functions inside one `<tool_call>` wrapper are each a call, strict and tolerant (candidates WD-43; vLLM
// #43713); a malformed later function still fails a strict region as a whole.
int test_functions_in_one_wrapper() {
    const auto contract = contract_from_definitions({tool_definition("first", Json::object()),
                                                     tool_definition("second", Json::object()),
                                                     tool_definition("third", Json::object())});
    const std::string text = "Working.\n<tool_call>\n<function=first>\n<parameter=a>\n1\n</parameter>\n</function>\n"
                             "<function=second>\n</function>\n"
                             "<function=third>\n<parameter=c>\nthree\n</parameter>\n</function>\n</tool_call>";
    int failures = 0;
    for (const bool tolerant : {false, true}) {
        const auto parsed = parse(text, 64, *contract, tolerant);
        failures += check(parsed.is_tool_call_response && parsed.content == "Working." &&
                              parsed.tool_calls.size() == 3 && parsed.tool_calls[0].name == "first" &&
                              parsed.tool_calls[1].name == "second" && parsed.tool_calls[2].name == "third" &&
                              parsed.diagnostics.fallback_reason == infernix::ToolCallParseFallbackReason::None,
                          tolerant ? "tolerant: functions in one wrapper were not each a call"
                                   : "strict: functions in one wrapper were not each a call");
        if (parsed.tool_calls.size() == 3) {
            failures += check(Json::parse(parsed.tool_calls[0].arguments_json).at("a") == 1 &&
                                  Json::parse(parsed.tool_calls[1].arguments_json).empty() &&
                                  Json::parse(parsed.tool_calls[2].arguments_json).at("c") == "three",
                              "a call in a shared wrapper lost its arguments");
        }
    }
    // The second function's parameter opener lost its `>` (a value that only lacks its closer is
    // recovered: test_unclosed_values).
    const std::string broken = "<tool_call>\n<function=first>\n</function>\n<function=second>\n<parameter=x\n1\n"
                               "</parameter>\n</function>\n</tool_call>";
    failures += check_rejected(broken, *contract, infernix::ToolCallParseFallbackReason::MalformedStructure,
                               "strict: a malformed second function in one wrapper did not fail the region");
    const auto tolerant_broken = parse(broken, 64, *contract, true);
    failures += check(tolerant_broken.tool_calls.size() == 1 && tolerant_broken.tool_calls[0].name == "first",
                      "tolerant: the complete first function before a malformed one was not kept");
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

    // Markup the grammar could continue from stays ambiguous, so that call still falls back.
    failures += check_rejected(tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}}),
                               contract, Reason::MalformedStructure,
                               "unbalanced nested parameter open was silently repaired");
    // A closer followed by the call's own closers and then text is value text when a later closer
    // ends the call (SY-04): only that reading parses the whole output, and the constrained route
    // reads such a value the same way.
    {
        const std::string command = "echo '</parameter></function>'";
        const auto parsed         = parse(tool_call("bash", {{"command", command}}), 64, contract);
        failures += check(parsed.tool_calls.size() == 1 &&
                              Json::parse(parsed.tool_calls.front().arguments_json).at("command") == command,
                          "a value quoting the call's closing tags was cut or demoted");
    }

    // Tolerant truncation: a closer at the region end still closes the value, and a quoted closer
    // inside a value the budget cut stays part of the kept partial value.
    const std::string open  = std::string("<") + "parameter=command>\n";
    const std::string close = std::string("</") + "parameter>";
    const std::string head  = "<tool_call>\n<function=bash>\n" + open;
    const auto closed_cut =
        parse(head + "ls\n" + close + "\n", 64, contract, true, /*cut_short*/ true);
    const auto quoted_cut = parse(
        head + "echo '" + close + "' more", 64, contract, true, /*cut_short*/ true);
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

    // Tolerant: a whole call whose value quotes an unmatched opener is not a cut call. The value
    // keeps the opener and the call keeps its sibling, with no recovery flag; a call cut inside
    // such a value still keeps the value to the cut.
    for (const std::string command : {std::string("echo '<parameter=x>'"),
                                      std::string("use <parameter=NAME> to name the argument")}) {
        for (const bool sibling : {false, true}) {
            const std::string text =
                sibling ? tool_call("bash", {{"command", command}, {"timeout", "30"}})
                        : tool_call("bash", {{"command", command}});
            const auto parsed = parse(text, 64, contract, true);
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  parsed.diagnostics.fallback_reason == Reason::None,
                              "a tolerant whole call quoting an unmatched opener read as cut");
            if (parsed.tool_calls.size() != 1) { continue; }
            const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
            failures += check(args.at("command") == command,
                              "a tolerant value quoting an unmatched opener was not kept exactly");
            failures += check(!sibling || args.at("timeout") == 30,
                              "the parameter after a quoted unmatched opener was not parsed");
        }
    }
    const auto quoted_open_cut =
        parse(head + "echo '<parameter=x>' more", 64, contract, true, /*cut_short*/ true);
    failures += check(quoted_open_cut.is_tool_call_response && quoted_open_cut.tool_calls.size() == 1 &&
                          quoted_open_cut.diagnostics.fallback_reason == Reason::TruncatedTail,
                      "a tolerant cut call quoting an unmatched opener was not kept as a truncated tail");
    if (quoted_open_cut.tool_calls.size() == 1) {
        const Json args = Json::parse(quoted_open_cut.tool_calls.front().arguments_json);
        failures += check(args.at("command") == "echo '<parameter=x>' more",
                          "a tolerant cut value quoting an unmatched opener lost its tail");
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

// A multi-line edit's array argument with real line breaks, tabs and carriage returns inside its
// strings (Strata #4630) decodes to the array the model meant, whole and at every streamed
// chunking; a declared string keeps its raw text, and other invalid text stays a string.
int test_control_bytes_inside_declared_json_strings() {
    const auto contract = contract_for(
        "edit_file", Json{{"path", Json{{"type", "string"}}},
                          {"edits", Json{{"type", "array"}}},
                          {"options", Json{{"type", "object"}}},
                          {"broken", Json{{"type", "array"}}}});
    const std::string edits = "[{\"old\":\"line one\nline two\",\"new\":\"a\tb\r\nc\"}]";
    const std::string text  = tool_call("edit_file", {{"path", "notes\n.md"},
                                                      {"edits", edits},
                                                      {"options", "{\"note\":\"x\ny\",\"n\":1}"},
                                                      {"broken", "[\"a\nb\", oops]"}});
    int failures            = 0;
    const auto expect       = [&](const std::string& arguments, const std::string& how) {
        const Json args = Json::parse(arguments);
        failures += check(args.at("edits").is_array() && args.at("edits").at(0).at("old") == "line one\nline two" &&
                              args.at("edits").at(0).at("new") == "a\tb\r\nc",
                          how + ": the array with raw control bytes in its strings was not decoded");
        failures += check(args.at("options").is_object() && args.at("options").at("note") == "x\ny" &&
                              args.at("options").at("n") == 1,
                          how + ": the object with a raw line break in a string was not decoded");
        failures += check(args.at("path") == "notes\n.md", how + ": a declared string lost its raw text");
        failures += check(args.at("broken") == "[\"a\nb\", oops]", how + ": invalid JSON did not stay a string");
    };
    const auto whole = parse(text, 64, contract);
    failures += check(whole.is_tool_call_response && whole.tool_calls.size() == 1, "the call was not parsed");
    if (whole.tool_calls.size() == 1) { expect(whole.tool_calls.front().arguments_json, "whole"); }
    for (std::size_t width = 1; width <= 7; ++width) {
        fi::ToolCallOutputDecoder decoder(output_contract_for("edit_file", Json{{"path", Json{{"type", "string"}}},
                                                                                {"edits", Json{{"type", "array"}}},
                                                                                {"options", Json{{"type", "object"}}},
                                                                                {"broken", Json{{"type", "array"}}}}),
                                          64);
        for (std::size_t at = 0; at < text.size(); at += width) { (void)feed(decoder, text, at, width); }
        const auto terminal = decoder.finish();
        failures += check(terminal.tool_calls.size() == 1, "a streamed chunking lost the call");
        if (terminal.tool_calls.size() == 1) {
            expect(terminal.tool_calls.front().arguments_json, "streamed at width " + std::to_string(width));
        }
    }
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

int test_unknown_schema_types_keep_inference() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"enum_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                        Json{{"enum", Json::array({1, 2})}}})}}},
             {"empty_intersection", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                         {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"enum_anyof", "7.5"},
                                                     {"empty_intersection", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = parse(text, 64, contract);

    // A schema naming no type it can read keeps the value text's own type; a union whose branches
    // are all typed (an enum's values type it) and a contradictory intersection are type hints, so
    // their values are mismatches, kept for the client to judge.
    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 3,
                      "unknown schema types did not keep inference or typed hints lost a mismatch");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("enum_anyof") == 7.5 && args.at("empty_intersection") == "True",
                      "a mismatched typed value was not kept as written");
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
        const auto parsed = parse(text, 64, *contract, /*tolerant*/ true, /*cut_short*/ true);
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
    // When the model itself ended the turn (a stop token), a call whose function was closed keeps
    // its complete arguments, but one the end left before its function's closing tag may be missing
    // arguments and stays text (vLLM #65693).
    const auto stopped_tool_close = parse(truncated_tool_close, 64, *contract, true);
    failures += check(stopped_tool_close.is_tool_call_response && stopped_tool_close.tool_calls.size() == 1,
                      "a stop after a closed function lost the complete call");
    const auto stopped_function_close = parse(truncated_function_close, 64, *contract, true);
    failures += check(!stopped_function_close.is_tool_call_response &&
                          stopped_function_close.tool_calls.empty() &&
                          stopped_function_close.diagnostics.fallback_reason == Reason::TruncatedTail &&
                          stopped_function_close.content.find("<function=delete_file>") != std::string::npos,
                      "a stop before the function's closing tag ran the call instead of returning text");
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
    // A client that knows tools it does not declare (PromptOptions::open_tool_names: Claude Code's
    // deferred MCP tools, SY-05) gets the call even from the strict parser, not as a repair; a drifted
    // opener still needs a declared name.
    auto open        = std::make_shared<fi::ToolCallOutputContract>(*contract);
    open->open_names = true;
    const auto opened = parse(undeclared, 64, *open);
    failures += check(opened.is_tool_call_response && opened.tool_calls.size() == 1 &&
                          opened.tool_calls.front().name == "not_a_declared_tool" &&
                          opened.diagnostics.fallback_reason == Reason::None &&
                          !opened.diagnostics.tolerant_recovered &&
                          Json::parse(opened.tool_calls.front().arguments_json).at("filePath") == "/tmp/out.js",
                      "open names did not keep the undeclared-name call");
    const auto drifted = parse("<tool_call>\n<=not_a_declared_tool>\n</function>\n</tool_call>", 64, *open, true);
    failures += check(!drifted.is_tool_call_response, "open names read an undeclared drifted opener as a call");

    const std::string value_cut = "Now\n"
                                  "<tool_call>\n"
                                  "<function=delete_file>\n"
                                  + open_tag + "/tmp/out";
    const auto cut_tolerant = parse(value_cut, 64, *contract, true, /*cut_short*/ true);
    failures += check(cut_tolerant.is_tool_call_response && cut_tolerant.tool_calls.size() == 1 &&
                          cut_tolerant.tool_calls.front().name == "delete_file",
                      "tolerant mode did not keep the value-cut call");
    // The same cut at the model's own stop token (vLLM #65693: a 44,271-character command ended at
    // the end-of-message token) is not a call: the turn returns it as text, flagged.
    const auto cut_stopped = parse(value_cut, 64, *contract, true);
    failures += check(!cut_stopped.is_tool_call_response && cut_stopped.tool_calls.empty() &&
                          cut_stopped.diagnostics.fallback_reason == Reason::TruncatedTail &&
                          !cut_stopped.diagnostics.tolerant_recovered,
                      "tolerant mode kept a call its own stop token cut inside a value");
    {
        fi::ToolCallOutputDecoder stopped(contract, 64, /*tolerant*/ true);
        std::string visible = feed(stopped, value_cut);
        const auto at_stop  = stopped.finish(infernix::FinishReason::StopToken);
        visible += at_stop.content;
        failures += check(at_stop.tool_calls.empty() && visible == value_cut,
                          "the decoder ran a call cut by a stop token instead of returning its text");
        fi::ToolCallOutputDecoder budget(contract, 64, /*tolerant*/ true);
        (void)feed(budget, value_cut);
        failures += check(budget.finish(infernix::FinishReason::OutputLimit).tool_calls.size() == 1,
                          "the decoder dropped a call the output budget cut");
        fi::ToolCallOutputDecoder by_string(contract, 64, /*tolerant*/ true);
        (void)feed(by_string, value_cut);
        failures += check(by_string.finish(infernix::FinishReason::StopString).tool_calls.empty(),
                          "the decoder ran a call cut by a stop string");
    }
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

// A call opener inside Markdown code of the answer (CommonMark's fences, indented code blocks and
// code spans) is a quoted example: it stays text, with no byte lost, while calls at the top level
// still fire, on the free and the constrained route alike. Only the text before an opener decides,
// so every streamed chunking agrees with the whole output.
int test_markdown_quoted_calls() {
    const std::vector<std::string> definitions = {
        tool_definition("write", Json{{"path", Json{{"type", "string"}}}})};
    const auto free_contract = contract_from_definitions(definitions);
    const auto constrained   = fi::select_tool_call_contract(free_contract, infernix::ToolChoice{});
    const auto call          = tool_call("write", {{"path", "/tmp/a.txt"}});
    // The call with `prefix` before each of its lines: indented four columns (an indented code block), or
    // inside a container (a quote's `> `, a list item's indentation).
    const auto lines_after = [&](const std::string& prefix) {
        std::string text = prefix + call;
        for (std::size_t at = text.find('\n'); at != std::string::npos; at = text.find('\n', at + 1)) {
            text.insert(at + 1, prefix);
        }
        return text;
    };
    const std::string indented = lines_after("    ");
    // (calls, content) of one output.
    using Outcome  = std::pair<std::size_t, std::string>;
    using Contract = std::shared_ptr<const fi::ToolCallOutputContract>;
    const auto run = [&](const Contract& contract, const std::string& text, bool tolerant, std::size_t width) {
        fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
        std::string content;
        for (std::size_t at = 0; at < text.size(); at += width) {
            content += feed(decoder, text, at, width);
        }
        auto terminal = decoder.finish();
        return Outcome(terminal.tool_calls.size(), content + terminal.content);
    };
    int failures       = 0;
    const auto outcome = [&](const Contract& contract, const std::string& text, bool tolerant) {
        const auto whole = parse(text, 64, *contract, tolerant);
        Outcome reference(whole.tool_calls.size(), whole.content);
        const std::vector<std::size_t> widths{1, 2, 3, 5, 7, 64, text.size()};
        for (const std::size_t width : widths) {
            if (run(contract, text, tolerant, width) != reference) {
                failures += fail("a streamed chunking of width " + std::to_string(width) +
                                 " disagrees with the whole output: " + text.substr(0, 40));
            }
        }
        return reference;
    };
    const std::vector<std::string> quoted{
        "Format:\n```xml\n" + call + "\n```\nDone.", "Format:\n~~~\n" + call + "\n~~~\nDone.",
        "Use `" + call + "` like this.", "```\n" + call,
        // A stray backtick on the call's own line still opens a span there.
        "Let me use `run_commands " + call,
        "Text\n\n```python\nx = 1\n```\n```\n" + call,
        // A shorter run, or a run with an info string, does not close a fence (Strata #4702).
        "````markdown\n```xml\n" + call + "\n```\n````\nDone.", "```\n```python\n" + call + "\n```",
        // An indented code block after a blank line or a heading (Strata #4703). When only the opener's
        // line is indented, the bare `<function=...>` after the quoted token is its wrapper's text.
        "Example:\n\n" + indented + "\n\nDone.", "Example:\n\n    " + call + "\n\nDone.", "# Format\n    " + call,
        // A code span closes only on a run of its own length.
        "Use ``a`" + call + "`` like this.",
        // Code inside containers: a fence in a block quote at any depth, in a quote inside a list item, or
        // in an item whose content starts four columns in; indented code four columns past an item's
        // content or a quote's.
        "> Format:\n> ```xml\n" + lines_after("> ") + "\n> ```\nDone.",
        "> > ~~~\n" + lines_after("> > ") + "\n> > ~~~",
        ">>>>>```\n" + lines_after(">>>>>"),
        "- Example:\n  > ```\n" + lines_after("  > ") + "\n  > ```",
        "10. Step:\n    ```xml\n" + lines_after("    ") + "\n    ```\nDone.",
        "- Step:\n\n      " + call,
        "> Example:\n>\n>     " + call};
    const std::vector<std::string> real{call,
                                        "Let me do it.\n\n" + call,
                                        "Let me do it. " + call,
                                        "```\ncode\n```\n" + call,
                                        "`a` and `b` " + call,
                                        "~~~\nx\n~~~\n\n" + call,
                                        "```\nx\n`````\n" + call,
                                        // Right after text an indented line continues the paragraph.
                                        "Example:\n    " + call,
                                        // Indented four columns, a run of backticks is code, not a fence.
                                        "Text\n\n    ```\n" + call,
                                        "Use ``a`b`` then " + call,
                                        "Run \\` first, then " + call,
                                        // A span left open on an earlier line of the paragraph does not
                                        // reach a call at a line's start (Strata #4923).
                                        "Let me use `run_commands to check.\n" + call,
                                        "Use `x`, then `\n   " + call,
                                        // A list item's continuation paragraph, four columns in after a
                                        // blank line, is text of the item, not code.
                                        "- Step one.\n\n    " + call,
                                        "- Step one.\n\n    Now run it: " + call,
                                        "1. Step:\n\n   " + call,
                                        // A fence ends with its container; a line without the quote's `>`
                                        // or the item's indentation continues their paragraph (laziness).
                                        "> Quote:\n> ```\n> x\n> ```\n\n" + call,
                                        "> ```\n> x\n" + call,
                                        "- ```\n  x\n" + call,
                                        "> Note:\n" + call,
                                        "- item\n" + call,
                                        // A thematic break is no list item.
                                        "Text\n- - -\n" + call};
    const std::vector<std::pair<Contract, bool>> routes{{free_contract, false}, {free_contract, true}, {constrained, false}};
    for (const auto& [contract, tolerant] : routes) {
        const std::string route = contract->constrained ? " (constrained)" : tolerant ? " (tolerant)" : " (strict)";
        for (const std::string& body : quoted) {
            const Outcome got = outcome(contract, body, tolerant);
            failures += check(got.first == 0 && got.second == body,
                              "a call quoted in Markdown code became a call or lost bytes" + route + ": " +
                                  body.substr(0, 30));
        }
        for (const std::string& body : real) {
            failures += check(outcome(contract, body, tolerant).first == 1,
                              "a top-level call after closed Markdown code did not fire" + route + ": " +
                                  body.substr(0, 30));
        }
        failures += check(outcome(contract, call + "\n" + call, tolerant).first == 2,
                          "two top-level calls did not both fire" + route);
        // A quoted example before the real call stays text; the call still fires.
        const std::string mixed = "Example:\n```\n" + call + "\n```\nNow for real.\n\n" + call;
        const Outcome both      = outcome(contract, mixed, tolerant);
        failures += check(both.first == 1 && both.second.starts_with("Example:\n```\n<tool_call>"),
                          "a quoted example before a real call was not kept as text" + route);
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
        // Closing lines that end the output close the call only at its end.
        failures += check(streamed.calls.size() == 1 && streamed.complete.size() == 1 && !streamed.complete[0],
                          "a call completed before the token after its closing lines");
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        streamed.take(decoder);
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

// A non-strict parameter's type hint follows a local `$ref`, the values of an `enum` or `const`,
// and a one-branch `allOf` (as Pydantic writes a described enum field): a string-valued one keeps
// every value as text. A reference that resolves to nothing, or only to itself, gives no hint, so
// the value text decides its type. The region parser (free output) and the constrained stream
// agree byte for byte.
int test_reference_and_value_type_hints() {
    const Json parameters = {
        {"type", "object"},
        {"$defs",
         {{"Status", {{"type", "string"}, {"enum", Json::array({"1", "2"})}}},
          {"Loop", {{"$ref", "#/$defs/Loop"}}}}},
        {"properties",
         {{"by_ref", {{"$ref", "#/$defs/Status"}}},
          {"enum_text", {{"enum", Json::array({"1", "2"})}}},
          {"constant", {{"const", "x"}}},
          {"wrapped",
           {{"allOf", Json::array({Json{{"$ref", "#/$defs/Status"}}})}, {"description", "state"}}},
          {"unresolved", {{"$ref", "#/$defs/Missing"}}},
          {"cyclic", {{"$ref", "#/$defs/Loop"}}}}}};
    const auto declarations = contract_from_definitions(
        {Json{{"type", "function"},
              {"function", {{"name", "configure"}, {"parameters", parameters}}}}
             .dump()});
    const auto constrained = fi::select_tool_call_contract(declarations, infernix::ToolChoice{});

    int failures = 0;
    failures += check(constrained && constrained->constrained,
                      "default tool choice did not constrain the calls");
    if (!constrained) { return failures; }
    const std::vector<std::string_view> hinted   = {"by_ref", "enum_text", "constant", "wrapped"};
    const std::vector<std::string_view> unhinted = {"unresolved", "cyclic"};
    // Each value, and its JSON as the unhinted parameters infer it from the text.
    const std::vector<std::pair<std::string_view, std::string_view>> values = {
        {"123", "123"}, {"true", "true"}, {"null", "null"}, {"007", "\"007\""}, {"abc", "\"abc\""}};
    for (const auto& [value, inferred] : values) {
        std::string text     = "<tool_call>\n<function=configure>\n";
        std::string expected = "{";
        for (const bool typed : {true, false}) {
            for (const std::string_view name : typed ? hinted : unhinted) {
                text += "<parameter=" + std::string(name) + ">\n" + std::string(value) +
                        "\n</parameter>\n";
                if (expected.size() > 1) { expected += ','; }
                expected += "\"" + std::string(name) + "\":" +
                            (typed ? Json(std::string(value)).dump() : std::string(inferred));
            }
        }
        text += "</function>\n</tool_call>";
        expected += '}';

        const auto parsed = parse(text, 64, *declarations);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().arguments_json == expected,
                          "a referenced or enum-typed value was retyped: " + std::string(value));

        fi::ToolCallOutputDecoder decoder(constrained, 64);
        StreamedCalls streamed;
        for (std::size_t at = 0; at < text.size(); ++at) {
            (void)feed(decoder, text, at, 1);
            streamed.take(decoder);
        }
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        streamed.take(decoder);
        failures += check(terminal.tool_calls.size() == 1 && streamed.calls.size() == 1 &&
                              terminal.tool_calls.front().arguments_json == expected &&
                              streamed.calls.front().arguments_json == expected,
                          "streamed arguments differ from the parsed ones: " + std::string(value));
    }
    return failures;
}

// A non-strict tool whose parameters are a root `oneOf` / `anyOf` of object variants (a zod
// discriminated union, Codex's `automation_update`; Basalt #208) declares its variants' properties:
// a string-typed one keeps `2026`, `true` and `null` as text. A name the variants type differently
// takes the union of their types (with a string among them it stays text); a variant that leaves it
// untyped, or a branch that cannot be read (a remote reference), leaves the value text to decide, as
// for an undeclared name. The free and the constrained route agree byte for byte.
int test_root_union_type_hints() {
    const Json zod = {
        {"type", "object"},
        {"properties", Json::object()},
        {"$defs",
         {{"Create", {{"type", "object"},
                      {"properties", {{"mode", {{"const", "create"}}}, {"name", {{"type", "string"}}},
                                      {"shared", {{"type", "string"}}}, {"loose", Json::object()}}}}},
          {"Update", {{"type", "object"},
                      {"properties", {{"mode", {{"const", "update"}}}, {"paused", {{"type", "boolean"}}},
                                      {"shared", {{"type", "integer"}}}, {"loose", {{"type", "string"}}}}}}}}},
        {"oneOf", Json::array({Json{{"$ref", "#/$defs/Create"}}, Json{{"$ref", "#/$defs/Update"}}})}};
    const Json remote = {
        {"type", "object"},
        {"anyOf", Json::array({Json{{"$ref", "https://example.com/schema.json"}},
                               Json{{"type", "object"}, {"properties", {{"name", {{"type", "string"}}}}}}})}};
    const auto declarations = contract_from_definitions(
        {Json{{"type", "function"}, {"function", {{"name", "automate"}, {"parameters", zod}}}}.dump(),
         Json{{"type", "function"}, {"function", {{"name", "fetch"}, {"parameters", remote}}}}.dump()});
    const auto constrained = fi::select_tool_call_contract(declarations, infernix::ToolChoice{});
    int failures           = 0;
    if (!constrained) { return check(false, "default tool choice did not constrain the calls"); }
    // Each call's arguments and the JSON they must decode to.
    const std::vector<std::tuple<std::string, std::vector<std::pair<std::string, std::string>>, std::string>> cases = {
        {"automate",
         {{"mode", "update"}, {"name", "2026"}, {"paused", "true"}, {"shared", "null"}, {"loose", "7"}},
         R"({"mode":"update","name":"2026","paused":true,"shared":"null","loose":7})"},
        {"automate", {{"name", "true"}, {"shared", "2026"}}, R"({"name":"true","shared":"2026"})"},
        {"fetch", {{"name", "2026"}}, R"({"name":2026})"}};
    for (const auto& [tool, arguments, expected] : cases) {
        std::string text = "<tool_call>\n<function=" + tool + ">\n";
        for (const auto& [name, value] : arguments) {
            text += "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
        }
        text += "</function>\n</tool_call>";
        const auto parsed = parse(text, 64, *declarations);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls.front().arguments_json == expected,
                          "a root union's arguments were typed wrongly: " + expected);
        fi::ToolCallOutputDecoder decoder(constrained, 64);
        (void)feed(decoder, text);
        const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
        failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().arguments_json == expected,
                          "the constrained route typed a root union's arguments differently: " + expected);
    }
    return failures;
}

// A generated value nested deeper than 256 arrays or objects stays text (a JSON string, counted as a schema mismatch),
// typed or not, so no call reaches a client as a tree too deep to serialize; 256 levels stay JSON (LE-14). A strict
// value streams as the model writes it, so the constrained decoder ends the request at the 257th level instead.
int test_argument_nesting_bound() {
    const auto nested = [](int levels) { return std::string(levels, '[') + std::string(levels, ']'); };
    const auto contract = contract_from_definitions(
        {tool_definition("store", Json{{"data", Json::object()}, {"list", Json{{"type", "array"}}}})});
    int failures = 0;
    for (const char* name : {"data", "list"}) {
        for (const int levels : {256, 257, 4000}) {
            const std::string value = nested(levels);
            const std::string text  = "<tool_call>\n<function=store>\n<parameter=" + std::string(name) + ">\n" + value +
                                     "\n</parameter>\n</function>\n</tool_call>";
            const auto parsed = parse(text, 64, *contract);
            const Json expected{{name, levels <= 256 ? Json::parse(value) : Json(value)}};
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                                  Json::parse(parsed.tool_calls.front().arguments_json) == expected &&
                                  (levels <= 256) == (parsed.diagnostics.schema_mismatch_arguments == 0),
                              "a value " + std::to_string(levels) + " levels deep in '" + name +
                                  "' was not kept as JSON up to 256 levels and as text beyond");
        }
    }
    const auto strict = fi::select_tool_call_contract(
        contract_from_definitions({Json{{"type", "function"},
                                        {"function", {{"name", "store"},
                                                      {"strict", true},
                                                      {"parameters", {{"type", "object"},
                                                                      {"properties", {{"list", {{"type", "array"}}}}},
                                                                      {"required", {"list"}},
                                                                      {"additionalProperties", false}}}}}}
                                       .dump()}),
        infernix::ToolChoice{});
    for (const int levels : {256, 257}) {
        fi::ToolCallOutputDecoder decoder(strict, 64);
        const std::string text = "<tool_call>\n<function=store>\n<parameter=list>\n" + nested(levels) +
                                 "\n</parameter>\n</function>\n</tool_call>";
        bool refused = false;
        try {
            (void)feed(decoder, text);
            (void)decoder.finish(infernix::FinishReason::StopToken);
        } catch (const std::length_error&) { refused = true; }
        failures += check(refused == (levels > 256),
                          "the constrained decoder did not stop a strict value at 257 levels, or stopped one at 256");
    }
    return failures;
}

// A file whose content documents the call format holds the call's own closing lines and then more
// text (Strata #4701): those lines are value text, so the file keeps every byte, on the free route
// (strict and tolerant, whole and at every chunking) and the constrained one, whose streamed steps
// concatenate to the call. Without a later closer, or with another call after the prose, a call
// still ends at its first closers, as before.
int test_quoted_call_closers() {
    const std::vector<std::string> definitions = {tool_definition(
        "write_file", Json{{"path", Json{{"type", "string"}}}, {"content", Json{{"type", "string"}}}})};
    const auto free_contract = contract_from_definitions(definitions);
    const auto constrained   = fi::select_tool_call_contract(free_contract, infernix::ToolChoice{});
    using Contract           = std::shared_ptr<const fi::ToolCallOutputContract>;
    const std::string document =
        "# Format\nA call ends with:\n</parameter>\n</function>\n</tool_call>\nThen the reply goes on.";
    const std::string call      = tool_call("write_file", {{"path", "/tmp/format.md"}, {"content", document}});
    const std::string arguments = Json{{"path", "/tmp/format.md"}, {"content", document}}.dump();
    int failures                = 0;
    const std::vector<std::pair<Contract, bool>> routes{{free_contract, false}, {free_contract, true}, {constrained, false}};
    for (const auto& [contract, tolerant] : routes) {
        const std::string route = contract->constrained ? " (constrained)" : tolerant ? " (tolerant)" : " (strict)";
        const std::string text  = "Writing it.\n" + call;
        if (!contract->constrained) {
            const auto whole = parse(text, 64, *contract, tolerant);
            failures += check(whole.tool_calls.size() == 1 && Json::parse(whole.tool_calls[0].arguments_json) ==
                                                                   Json::parse(arguments),
                              "the whole parse cut a file at its quoted closing lines" + route);
        }
        for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{7}, text.size()}) {
            fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
            StreamedCalls streamed;
            std::string visible;
            for (std::size_t at = 0; at < text.size(); at += width) {
                visible += feed(decoder, text, at, width);
                streamed.take(decoder);
            }
            const auto terminal = decoder.finish(infernix::FinishReason::StopToken);
            streamed.take(decoder);
            failures += check(terminal.tool_calls.size() == 1 &&
                                  Json::parse(terminal.tool_calls[0].arguments_json) == Json::parse(arguments) &&
                                  visible + terminal.content == "Writing it.",
                              "a file was cut at its quoted closing lines" + route + ", width " + std::to_string(width));
            if (contract->constrained && terminal.tool_calls.size() == 1) {
                failures += check(streamed.ordered && streamed.calls.size() == 1 && streamed.complete[0] &&
                                      streamed.calls[0].arguments_json == terminal.tool_calls[0].arguments_json,
                                  "the streamed steps of a file holding closing lines differ, width " + std::to_string(width));
            }
        }
    }
    // Prose after a call with no later closer: strict demotes it to text, tolerant keeps it cut there.
    const std::string plain   = tool_call("write_file", {{"path", "/tmp/a"}, {"content", "x"}});
    const std::string trailed = plain + "\nThat wrote the file.";
    failures += check(parse(trailed, 64, *free_contract).tool_calls.empty(), "strict kept a call with trailing prose");
    const auto kept = parse(trailed, 64, *free_contract, true);
    failures += check(kept.tool_calls.size() == 1 && Json::parse(kept.tool_calls[0].arguments_json)["content"] == "x",
                      "tolerant did not keep a call before trailing prose");
    // Another call after the prose: the first call's closers stay its closers (no value holds the
    // `<tool_call>` token), as before: strict takes the second call, tolerant the first.
    const std::string two = plain + "\nNext one.\n" + tool_call("write_file", {{"path", "/tmp/b"}, {"content", "y"}});
    const auto strict_two = parse(two, 64, *free_contract);
    failures += check(strict_two.tool_calls.size() == 1 &&
                          Json::parse(strict_two.tool_calls[0].arguments_json)["path"] == "/tmp/b",
                      "strict read prose between two calls differently");
    const auto tolerant_two = parse(two, 64, *free_contract, true);
    failures += check(tolerant_two.tool_calls.size() == 1 &&
                          Json::parse(tolerant_two.tool_calls[0].arguments_json)["content"] == "x",
                      "tolerant let a value swallow the next call");
    return failures;
}

// The terminal turn of a generated text on one route. The whole parse (free routes) and the streamed
// decoder at widths 1, 3, 7 and whole must agree (`consistent`), and a constrained call's streamed steps
// must concatenate to it.
struct Turn {
    std::vector<infernix::GeneratedToolCall> calls;
    infernix::ToolCallParseDiagnostics diagnostics;
    std::string content;
    bool consistent = true;

    [[nodiscard]] std::vector<Json> arguments() const {
        std::vector<Json> out;
        for (const auto& call : calls) { out.push_back(Json::parse(call.arguments_json)); }
        return out;
    }
};

enum class Route { Strict, Tolerant, Constrained };

std::string route_name(Route route) {
    return route == Route::Strict ? " (strict)" : route == Route::Tolerant ? " (tolerant)" : " (constrained)";
}

Turn turn_on(Route route, const std::shared_ptr<const fi::ToolCallOutputContract>& declarations, const std::string& text,
             infernix::FinishReason reason = infernix::FinishReason::StopToken) {
    const bool tolerant  = route == Route::Tolerant;
    const auto contract  = route == Route::Constrained
                               ? fi::select_tool_call_contract(declarations, infernix::ToolChoice{})
                               : declarations;
    const bool cut_short = reason == infernix::FinishReason::OutputLimit;
    std::optional<Turn> turn;
    const auto agree = [&](Turn next) {
        if (!turn) {
            turn = std::move(next);
            return;
        }
        bool same = next.calls.size() == turn->calls.size() && next.diagnostics == turn->diagnostics &&
                    next.content == turn->content && next.consistent;
        for (std::size_t i = 0; same && i < next.calls.size(); ++i) {
            same = next.calls[i].name == turn->calls[i].name &&
                   next.calls[i].arguments_json == turn->calls[i].arguments_json;
        }
        turn->consistent = turn->consistent && same;
    };
    if (route != Route::Constrained) {
        auto parsed = parse(text, 64, *contract, tolerant, cut_short);
        agree({.calls = std::move(parsed.tool_calls), .diagnostics = parsed.diagnostics, .content = parsed.content});
    }
    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{7}, text.size()}) {
        fi::ToolCallOutputDecoder decoder(contract, 64, tolerant);
        StreamedCalls streamed;
        std::string visible;
        for (std::size_t at = 0; at < text.size(); at += width) {
            visible += feed(decoder, text, at, width);
            streamed.take(decoder);
        }
        auto terminal = decoder.finish(reason);
        streamed.take(decoder);
        bool steps = true;
        if (route == Route::Constrained) {
            steps = streamed.ordered && streamed.calls.size() >= terminal.tool_calls.size();
            for (std::size_t i = 0; steps && i < terminal.tool_calls.size(); ++i) {
                steps = streamed.complete[i] && streamed.calls[i].arguments_json == terminal.tool_calls[i].arguments_json;
            }
        }
        agree({.calls       = std::move(terminal.tool_calls),
               .diagnostics = terminal.diagnostics,
               .content     = visible + terminal.content,
               .consistent  = steps});
    }
    return std::move(*turn);
}

// A string value that never closes (Strata #5756, strata-nvfp4 #24). The model skips the value's own
// closer after an inline nested pair (a file that documents the call format), or its closer loses the
// `>`. Strict: the region stays text. Tolerant: a closer on the line of a nested opener closes that pair,
// and a value no closer ends goes on to the call's next declared parameter or its closers (without a
// closer that lost its `>`), reported as recovered; streamed and whole agree, as does a budget stop
// after the call's closers. A quoted unmatched opener keeps SR-07's reading.
int test_unclosed_values() {
    using Reason            = infernix::ToolCallParseFallbackReason;
    const auto declarations = output_contract_for("edit_file", Json{{"path", Json{{"type", "string"}}},
                                                                    {"old_string", Json{{"type", "string"}}},
                                                                    {"new_string", Json{{"type", "string"}}}});
    const std::string head  = "<tool_call>\n<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n";
    const std::string tail  = "</function>\n</tool_call>";
    const std::string pair  = "<parameter=limit>25</parameter>";
    const Json path{{"path", "docs/tools.md"}};
    const auto with = [&](Json extra) {
        Json arguments = path;
        arguments.update(extra);
        return arguments;
    };
    struct Case {
        std::string label;
        std::string text;
        std::vector<Json> tolerant; // the tolerant calls' arguments
        bool recovered;             // reported as tolerant_recovered
        // The strict calls' arguments; nullopt: the strict parser returns the text.
        std::optional<std::vector<Json>> strict;
    };
    const std::vector<Case> cases{
        {"a nested pair, then the call's closers", head + "<parameter=new_string>\n" + pair + "\n" + tail,
         {with({{"new_string", pair}})}, true, std::nullopt},
        {"a nested pair, then the next declared parameter inline",
         head + "<parameter=new_string>\n" + pair + "\n<parameter=old_string>x</parameter>\n" + tail,
         {with({{"new_string", pair}, {"old_string", "x"}})}, true, std::nullopt},
        {"a nested pair, then the next declared parameter on lines of its own",
         head + "<parameter=new_string>\n" + pair + "\n<parameter=old_string>\nx\n</parameter>\n" + tail,
         {with({{"new_string", pair}, {"old_string", "x"}})}, true, std::nullopt},
        {"a nested pair and an undeclared inline pair, then the call's closers",
         head + "<parameter=new_string>\n" + pair + "\n<parameter=other>y</parameter>\n" + tail,
         {with({{"new_string", pair + "\n<parameter=other>y</parameter>"}})}, true, std::nullopt},
        // Strict: the first region fails, so the second call is the turn.
        {"a nested pair, then the call's closers and another call",
         head + "<parameter=new_string>\n" + pair + "\n" + tail + "\n" + tool_call("edit_file", {{"path", "b.md"}}),
         {with({{"new_string", pair}}), Json{{"path", "b.md"}}}, true, std::vector<Json>{Json{{"path", "b.md"}}}},
        {"a nested pair closed by the value's own closer", head + "<parameter=new_string>\n" + pair + "\n</parameter>\n" + tail,
         {with({{"new_string", pair}})}, false, std::vector<Json>{with({{"new_string", pair}})}},
        {"a value without its closer", head + "<parameter=new_string>\nhello\n" + tail, {with({{"new_string", "hello"}})},
         true, std::nullopt},
        {"a closer that lost its >", "<tool_call>\n<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter\n" + tail,
         {path}, true, std::nullopt},
        // SR-07: an opener on a line of its own, or in a value written on the opener's line, is quoted.
        {"a multi-line nested opener", head + "<parameter=new_string>\nExample:\n<parameter=limit>\n25\n</parameter>\n" + tail,
         {with({{"new_string", "Example:\n<parameter=limit>\n25"}})}, false, std::nullopt},
        {"an unframed value quoting an opener",
         head + "<parameter=new_string>echo '<parameter=x>'</parameter>\n" + tail,
         {with({{"new_string", "echo '<parameter=x>'"}})}, false, std::nullopt},
        {"a declared opener on a line of its own, no nested pair before it",
         head + "<parameter=new_string>\nTo set it:\n<parameter=old_string>\nand close it.\n</parameter>\n" + tail,
         {with({{"new_string", "To set it:\n<parameter=old_string>\nand close it."}})}, false, std::nullopt},
    };
    int failures = 0;
    for (const Case& c : cases) {
        for (const auto reason : {infernix::FinishReason::StopToken, infernix::FinishReason::OutputLimit}) {
            const std::string how = c.label + (reason == infernix::FinishReason::StopToken ? "" : ", budget stop");
            const Turn tolerant   = turn_on(Route::Tolerant, declarations, c.text, reason);
            failures += check(tolerant.consistent && tolerant.arguments() == c.tolerant &&
                                  tolerant.diagnostics.fallback_reason == Reason::None &&
                                  tolerant.diagnostics.tolerant_recovered == c.recovered,
                              "tolerant read of " + how + " changed");
        }
        const Turn strict = turn_on(Route::Strict, declarations, c.text);
        failures += check(strict.consistent && (c.strict ? strict.arguments() == *c.strict
                                                         : strict.calls.empty() && strict.content == c.text &&
                                                               strict.diagnostics.fallback_reason ==
                                                                   Reason::MalformedStructure),
                          "strict read of " + c.label + " changed");
    }
    // A parameter opener that lost its `>` (strata-nvfp4 #24 cases A and B) names no parameter: its header
    // would run to a later tag's `>` across lines. The region stays text on every free route.
    for (const std::string& text :
         {"<tool_call>\n<function=edit_file>\n<parameter=path\ndocs/tools.md\n</parameter>\n" + tail,
          head + "<parameter=new_string\nhello\n</parameter>\n" + tail}) {
        for (const auto reason : {infernix::FinishReason::StopToken, infernix::FinishReason::OutputLimit}) {
            for (const Route route : {Route::Strict, Route::Tolerant}) {
                const Turn turn = turn_on(route, declarations, text, reason);
                failures += check(turn.consistent && turn.calls.empty() && turn.content == text,
                                  "a parameter opener without its > was read as a parameter" + route_name(route));
            }
        }
    }
    return failures;
}

// A value the output budget cuts inside its closer keeps no fragment of it (ZD-42), whatever byte of
// `\n</parameter>` (or `\n</param>`) the cut follows, whole and streamed; a `<` that does not start a line
// stays value text.
int test_value_cut_inside_closer() {
    using Reason            = infernix::ToolCallParseFallbackReason;
    const auto declarations = output_contract_for("edit_file", Json{{"path", Json{{"type", "string"}}}});
    int failures            = 0;
    for (const std::string close : {std::string("</parameter>"), std::string("</param>")}) {
        const std::string open   = close == "</param>" ? "<param=path>" : "<parameter=path>";
        const std::string head   = "<tool_call>\n<function=edit_file>\n" + open + "\ndocs/tools.md";
        const std::string closer = "\n" + close;
        for (std::size_t n = 0; n <= closer.size(); ++n) {
            const std::string text = head + closer.substr(0, n);
            const Turn turn = turn_on(Route::Tolerant, declarations, text, infernix::FinishReason::OutputLimit);
            failures += check(turn.consistent && turn.arguments() == std::vector<Json>{Json{{"path", "docs/tools.md"}}} &&
                                  turn.diagnostics.fallback_reason == Reason::TruncatedTail &&
                                  turn.diagnostics.tolerant_recovered,
                              "a value cut after " + std::to_string(n) + " bytes of " + close + " kept a fragment");
        }
    }
    const Turn kept = turn_on(Route::Tolerant, declarations, "<tool_call>\n<function=edit_file>\n<parameter=path>\nx <",
                              infernix::FinishReason::OutputLimit);
    failures += check(kept.arguments() == std::vector<Json>{Json{{"path", "x <"}}},
                      "a cut value lost a `<` that does not start a line");
    return failures;
}

// A call whose only parameter is an `arguments` its tool does not declare, holding a JSON object, is
// that object (ZA-02, Strata #5804), byte for byte on every route, and counted; a declared `arguments`,
// a value that is not an object, a second parameter or an undeclared tool keep the parameter.
int test_lone_arguments_parameter() {
    const auto declarations = contract_from_definitions(
        {tool_definition("edit_file", Json{{"path", Json{{"type", "string"}}}, {"new_string", Json{{"type", "string"}}}}),
         tool_definition("configure", Json{{"arguments", Json{{"type", "object"}}}})});
    const std::string object = "{\"path\":\"a.md\", \"new_string\":\"x\\ny\"}";
    struct Case {
        std::string label;
        std::string text;
        std::vector<std::string> arguments; // the calls' arguments JSON, exactly
        std::uint32_t unwrapped;
        std::uint32_t mismatched;
    };
    const std::vector<Case> cases{
        {"an undeclared lone arguments object", tool_call("edit_file", {{"arguments", object}}), {object}, 1, 0},
        {"one with a raw line break in a string", tool_call("edit_file", {{"arguments", "{\"new_string\":\"a\nb\"}"}}),
         {"{\"new_string\":\"a\\nb\"}"}, 1, 0},
        {"two calls", tool_call("edit_file", {{"arguments", "{}"}}) + "\n" + tool_call("edit_file", {{"arguments", object}}),
         {"{}", object}, 2, 0},
        {"a declared arguments", tool_call("configure", {{"arguments", "{\"a\":1}"}}), {"{\"arguments\":{\"a\":1}}"}, 0, 0},
        {"an array", tool_call("edit_file", {{"arguments", "[1,2]"}}), {"{\"arguments\":[1,2]}"}, 0, 1},
        {"text", tool_call("edit_file", {{"arguments", "not json"}}), {"{\"arguments\":\"not json\"}"}, 0, 1},
        {"a second parameter", tool_call("edit_file", {{"arguments", "{\"a\":1}"}, {"path", "b.md"}}),
         {"{\"arguments\":{\"a\":1},\"path\":\"b.md\"}"}, 0, 1},
    };
    int failures = 0;
    for (const Case& c : cases) {
        for (const Route route : {Route::Strict, Route::Tolerant, Route::Constrained}) {
            const Turn turn = turn_on(route, declarations, c.text);
            bool exact      = turn.calls.size() == c.arguments.size();
            for (std::size_t i = 0; exact && i < turn.calls.size(); ++i) {
                exact = turn.calls[i].arguments_json == c.arguments[i];
            }
            failures += check(turn.consistent && exact && turn.diagnostics.arguments_unwrapped == c.unwrapped &&
                                  turn.diagnostics.schema_mismatch_arguments == c.mismatched,
                              "lone arguments: " + c.label + route_name(route));
        }
    }
    // A tool the request does not declare (kept by tolerant mode): its parameters are unknown.
    const Turn undeclared = turn_on(Route::Tolerant, declarations, tool_call("unknown_tool", {{"arguments", "{\"a\":1}"}}));
    failures += check(undeclared.calls.size() == 1 && undeclared.calls[0].arguments_json == "{\"arguments\":{\"a\":1}}" &&
                          undeclared.diagnostics.arguments_unwrapped == 0,
                      "an undeclared tool's arguments parameter was unwrapped");
    return failures;
}

// Python's `None` (any case, as booleans) is null where a non-string declared type admits null (ZH-06);
// a string type keeps the text, and a type without null keeps it as a mismatched string.
int test_python_none() {
    const auto declarations = output_contract_for(
        "configure", Json{{"count", Json{{"type", Json::array({"integer", "null"})}}},
                          {"flag", Json{{"type", Json::array({"boolean", "null"})}}},
                          {"ratio", Json{{"type", Json::array({"number", "null"})}}},
                          {"payload", Json{{"type", Json::array({"object", "null"})}}},
                          {"items", Json{{"anyOf", Json::array({Json{{"type", "array"}}, Json{{"type", "null"}}})}}},
                          {"unset", Json{{"type", "null"}}},
                          {"plain", Json{{"type", "integer"}}},
                          {"label", Json{{"type", Json::array({"string", "null"})}}},
                          {"name", Json{{"type", "string"}}},
                          {"switch", Json{{"type", Json::array({"boolean", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "None"},
                                                     {"flag", "none"},
                                                     {"ratio", "NONE"},
                                                     {"payload", "None"},
                                                     {"items", "None"},
                                                     {"unset", "None"},
                                                     {"plain", "None"},
                                                     {"label", "None"},
                                                     {"name", "None"},
                                                     {"switch", "True"}});
    const Json expected{{"count", nullptr},  {"flag", nullptr},   {"ratio", nullptr}, {"payload", nullptr},
                        {"items", nullptr},  {"unset", nullptr},  {"plain", "None"},  {"label", "None"},
                        {"name", "None"},    {"switch", true}};
    int failures = 0;
    for (const Route route : {Route::Strict, Route::Tolerant, Route::Constrained}) {
        const Turn turn = turn_on(route, declarations, text);
        failures += check(turn.consistent && turn.arguments() == std::vector<Json>{expected} &&
                              turn.diagnostics.schema_mismatch_arguments == 1,
                          "None was not read as null where the type admits it" + route_name(route));
    }
    return failures;
}

// An array or object value refused only for a quote between two word characters inside a string (a
// Hebrew abbreviation, 5"x7) is read with it escaped (ZD-43), together with raw control bytes, on every
// route; valid JSON keeps its text, and other quotes, a repair of a type the schema does not admit, or an
// untyped value stay strings.
int test_inner_quote_repair() {
    const std::string tsadi_he = "\xd7\xa6\xd7\x94", lamed = "\xd7\x9c";
    const auto declarations = output_contract_for(
        "configure", Json{{"record", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"note", Json{{"type", Json::array({"object", "null"})}}},
                          {"valid", Json{{"type", "object"}}},
                          {"spaced", Json{{"type", "object"}}},
                          {"list", Json{{"type", "array"}}},
                          {"untyped", Json::object()},
                          {"name", Json{{"type", "string"}}}});
    const std::string valid = "{\"q\":\"say \\\"hi\\\"\"}";
    const std::string text  = tool_call("configure", {{"record", "{\"unit\":\"" + tsadi_he + "\"" + lamed + "\"}"},
                                                      {"items", "[\"5\"x7 frame\", \"ok\"]"},
                                                      {"note", "{\"t\":\"a\"b\nc\"}"},
                                                      {"valid", valid},
                                                      {"spaced", "{\"a\":\"x\" y\"}"},
                                                      {"list", "{\"a\":\"b\"c\"}"},
                                                      {"untyped", "{\"a\":\"b\"c\"}"},
                                                      {"name", "5\"x7"}});
    const Json expected{{"record", Json{{"unit", tsadi_he + "\"" + lamed}}},
                        {"items", Json::array({"5\"x7 frame", "ok"})},
                        {"note", Json{{"t", "a\"b\nc"}}},
                        {"valid", Json{{"q", "say \"hi\""}}},
                        {"spaced", "{\"a\":\"x\" y\"}"},
                        {"list", "{\"a\":\"b\"c\"}"},
                        {"untyped", "{\"a\":\"b\"c\"}"},
                        {"name", "5\"x7"}};
    int failures = 0;
    for (const Route route : {Route::Strict, Route::Tolerant, Route::Constrained}) {
        const Turn turn = turn_on(route, declarations, text);
        failures += check(turn.consistent && turn.arguments() == std::vector<Json>{expected} &&
                              turn.diagnostics.schema_mismatch_arguments == 2 &&
                              turn.calls.size() == 1 &&
                              turn.calls[0].arguments_json.find("\"valid\":" + valid) != std::string::npos,
                          "an unescaped inner quote was not repaired as specified" + route_name(route));
    }
    return failures;
}

int run_tests() {
    int failures = 0;
    failures += test_constrained_call_streaming();
    failures += test_opener_token_identity();
    failures += test_free_tool_continuation();
    failures += test_untyped_parameters();
    failures += test_invalid_utf8_parameter_value();
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_multiple_calls();
    failures += test_functions_in_one_wrapper();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_parameter_delimiters_in_values();
    failures += test_declared_json_types();
    failures += test_control_bytes_inside_declared_json_strings();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unknown_schema_types_keep_inference();
    failures += test_reference_and_value_type_hints();
    failures += test_root_union_type_hints();
    failures += test_argument_nesting_bound();
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
    failures += test_quoted_call_closers();
    failures += test_unclosed_values();
    failures += test_value_cut_inside_closer();
    failures += test_lone_arguments_parameter();
    failures += test_python_none();
    failures += test_inner_quote_repair();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

// `--probe-parse` (tests/models/qwen3_5/test_response_template_oracle.py): one JSON request a line,
// {"text": ..., "tools": {"name": ["parameter", ...]}}, every parameter a declared string so values keep
// their text; one JSON line back: {"tool_call_response", "content", "calls": [{"name", "arguments"}]}, or
// {"error"}.
int probe_parse() {
    std::string line;
    while (std::getline(std::cin, line)) {
        Json response;
        try {
            const Json input = Json::parse(line);
            std::vector<std::string> definitions;
            for (const auto& [name, parameters] : input.at("tools").items()) {
                Json properties = Json::object();
                for (const auto& parameter : parameters) {
                    properties[parameter.get<std::string>()] = Json{{"type", "string"}};
                }
                definitions.push_back(tool_definition(name, std::move(properties)));
            }
            const auto contract = contract_from_definitions(definitions);
            const auto parsed   = parse(input.at("text").get<std::string>(), 64, *contract);
            Json calls          = Json::array();
            for (const auto& call : parsed.tool_calls) {
                calls.push_back(Json{{"name", call.name}, {"arguments", Json::parse(call.arguments_json)}});
            }
            response = Json{{"tool_call_response", parsed.is_tool_call_response},
                            {"content", parsed.content},
                            {"calls", std::move(calls)}};
        } catch (const std::exception& error) { response = Json{{"error", error.what()}}; }
        std::cout << response.dump() << '\n' << std::flush;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--probe-parse") { return probe_parse(); }
    return run_tests();
}
