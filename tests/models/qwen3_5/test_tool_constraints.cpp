#include "models/qwen3_5/frontend/tool_contract.h"
#include "models/qwen3_5/frontend/tool_grammar.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/grammar.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
namespace frontend = infernix::models::qwen3_5::frontend;
using Json         = nlohmann::ordered_json;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Byte tokens 0-255, EOS 256 and the model's `<tool_call>` token.
constexpr std::int32_t kEos      = 256;
constexpr std::int32_t kToolCall = 257;

infernix::text::GrammarCompiler compiler() {
    std::vector<std::string> tokens;
    for (int i = 0; i < 256; ++i) tokens.emplace_back(1, static_cast<char>(i));
    tokens.emplace_back();
    tokens.emplace_back(frontend::kToolCallOpen);
    return {std::move(tokens), {kEos}, 32 * 1024 * 1024};
}

// The offsets of the `<tool_call>` spellings in generated text: the model writes each as its token.
std::vector<std::size_t> openers(std::string_view text) {
    std::vector<std::size_t> found;
    for (auto at = text.find(frontend::kToolCallOpen); at != std::string_view::npos;
         at = text.find(frontend::kToolCallOpen, at + 1))
        found.push_back(at);
    return found;
}

// The tokens of generated text: each `<tool_call>` spelling as the model's token unless `spelled`,
// every other byte as its byte token.
std::vector<std::int32_t> tokens(std::string_view text, bool spelled = false) {
    std::vector<std::int32_t> result;
    for (std::size_t at = 0; at < text.size();) {
        if (!spelled && text.substr(at).starts_with(frontend::kToolCallOpen)) {
            result.push_back(kToolCall);
            at += frontend::kToolCallOpen.size();
        } else {
            result.push_back(static_cast<unsigned char>(text[at++]));
        }
    }
    return result;
}

// Feeds bytes [begin, begin + size) of generated text, with the `<tool_call>` tokens starting there.
std::string feed(frontend::ToolCallOutputDecoder& decoder, std::string_view text,
                 std::size_t begin = 0, std::size_t size = std::string_view::npos) {
    const auto piece = text.substr(begin, size);
    std::vector<std::size_t> starts;
    for (const auto at : openers(text))
        if (at >= begin && at - begin < piece.size()) starts.push_back(at - begin);
    return decoder.feed({piece, starts});
}

auto contract(const Json& schema, bool strict, infernix::ToolChoice choice = {}) {
    std::vector<std::string> definitions{
        Json{{"type", "function"},
             {"function", {{"name", "call"}, {"parameters", schema}, {"strict", strict}}}}
            .dump()};
    return frontend::select_tool_call_contract(
        frontend::build_tool_call_output_contract(definitions), choice);
}

bool accepts_tokens(infernix::text::GrammarCompiler& compiler,
                    const frontend::ToolCallOutputContract& contract,
                    const std::vector<std::int32_t>& value) {
    auto matcher = frontend::compile_tool_grammar(compiler, contract, kToolCall, {}, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    for (const std::int32_t token : value) {
        if (matcher->masks({}, words) || !(words[token / 32] & (1u << (token % 32)))) return false;
        matcher->accept(token);
        matcher->confirm();
    }
    return matcher->masks({}, words) == 0 && (words[kEos / 32] & (1u << (kEos % 32)));
}

bool accepts(infernix::text::GrammarCompiler& compiler,
             const frontend::ToolCallOutputContract& contract, std::string_view value) {
    return accepts_tokens(compiler, contract, tokens(value));
}

std::string call(std::string_view value, std::string_view name = "call") {
    return "<tool_call>\n<function=" + std::string(name) + ">\n<parameter=x>\n" +
           std::string(value) + "\n</parameter>\n</function>\n</tool_call>";
}

Json schema(const Json& value) {
    return {{"type", "object"},
            {"properties", {{"x", value}}},
            {"required", {"x"}},
            {"additionalProperties", false}};
}

void basic_contracts(infernix::text::GrammarCompiler& compiled) {
    const Json source{
        {"type", "object"},
        {"properties", {{"a", {{"type", "string"}}}, {"b", {{"type", "integer"}, {"minimum", 5}}}}},
        {"required", {"a", "b"}},
        {"additionalProperties", true}};
    const auto basic = contract(source, false);
    require(basic->constrained, "ordinary tools did not enable basic constraints");
    const std::string reordered =
        "<tool_call>\n<function=call>\n<parameter=b>\n1\n</parameter>\n"
        "<parameter=a>\n007\n</parameter>\n<parameter=extra-key.键>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>";
    require(accepts(compiled, *basic, reordered),
            "basic closed or reordered an open parameter list");
    frontend::ToolCallOutputDecoder decoded(basic, 64);
    (void)feed(decoded, reordered);
    const auto arguments = Json::parse(decoded.finish().tool_calls.at(0).arguments_json);
    require(arguments == Json{{"b", 1}, {"a", "007"}, {"extra-key.键", true}},
            "non-strict value normalization changed");
    require(accepts(compiled, *basic, "An ordinary answer."), "default auto forced a call");
    require(
        !accepts(compiled, *basic, "<tool_call>\n<function=missing>\n</function>\n</tool_call>"),
        "default auto admitted an undeclared function");

    for (const Json& complex :
         {Json{{"anyOf", {{{"type", "object"}}, {{"type", "object"}, {"required", {"x"}}}}}},
          Json{{"$ref", "https://example.test/schema"}},
          Json{{"type", "object"}, {"patternProperties", {{".*", {{"type", "string"}}}}}},
          schema({{"type", {"string", "null"}}, {"format", "uri"}})}) {
        const auto bound = contract(complex, false);
        require(accepts(compiled, *bound, call("hello")),
                "non-strict schema reached strict validation or closed the parameter list");
    }

    // A call names each parameter once (the grammar's companion), so a streamed call never repeats a
    // key; a name extending a used one, and the same name in the next call, stay open.
    const auto integer      = contract(schema({{"type", "integer"}}), false);
    const std::string first = "<tool_call>\n<function=call>\n<parameter=x>\n1\n</parameter>\n";
    require(!accepts(compiled, *integer, first + "<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>"),
            "basic arguments admitted a repeated parameter name");
    require(accepts(compiled, *integer,
                    first + "<parameter=xy>\n2\n</parameter>\n</function>\n</tool_call>\n" + call("3")),
            "the parameter-name rule refused a new name or the next call's");
    {
        // Lookahead through a draft that repeats the name masks the `>` that would complete it and
        // leaves the session where it was; accepting that `>` is refused.
        auto matcher = frontend::compile_tool_grammar(compiled, *integer, kToolCall, {}, {});
        for (const std::int32_t token : tokens(first + "<parameter=")) {
            matcher->accept(token);
            matcher->confirm();
        }
        const std::vector<std::int32_t> drafts{'x', '>'};
        std::vector<std::uint32_t> words(3 * matcher->mask_words());
        (void)matcher->masks(drafts, words);
        const auto after_x = std::span(words).subspan(matcher->mask_words(), matcher->mask_words());
        require((after_x['>' / 32] & (1u << ('>' % 32))) == 0 && (after_x['y' / 32] & (1u << ('y' % 32))) != 0,
                "lookahead did not mask the repeated name's end");
        std::vector<std::uint32_t> now(matcher->mask_words());
        (void)matcher->masks({}, now);
        require((now['x' / 32] & (1u << ('x' % 32))) != 0, "lookahead moved the companion's state");
        matcher->accept('x');
        bool refused = false;
        try {
            matcher->accept('>');
        } catch (const std::logic_error&) { refused = true; }
        require(refused, "a repeated parameter name was accepted");
        matcher->discard();
    }

    // These requests share the same compiled envelope but have different value normalization.
    const auto string = contract(schema({{"type", "string"}}), false);
    for (const auto& bound : {integer, string}) {
        require(accepts(compiled, *bound, call("7")),
                "shared basic grammar changed with value types");
        frontend::ToolCallOutputDecoder decoder(bound, 64);
        (void)feed(decoder, call("7"));
        const auto args = Json::parse(decoder.finish().tool_calls.at(0).arguments_json);
        require(args["x"] == (bound == string ? Json("7") : Json(7)),
                "compiled grammar leaked another request's argument types");
    }

    const auto two = reordered + "\n" + call("later");
    for (std::size_t cut = 0; cut < two.size(); ++cut) {
        frontend::ToolCallOutputDecoder decoder(basic, 64);
        (void)feed(decoder, std::string_view(two).substr(0, cut));
        const auto result = decoder.finish(infernix::FinishReason::OutputLimit);
        require(result.tool_calls.size() == (cut >= reordered.size() ? 1 : 0),
                "non-strict truncation lost a completed call");
    }
    infernix::ToolChoice automatic;
    automatic.constraints = infernix::ToolConstraintMode::Automatic;
    require(!contract(source, false, automatic)->constrained,
            "explicit auto ignored the free route");
    automatic.mode = infernix::ToolChoiceMode::Required;
    require(contract(source, false, automatic)->constrained, "auto relaxed a required choice");
    require(contract(schema({{"type", "integer"}}), true, automatic)->constrained,
            "auto relaxed strict arguments");
}

void selected_contracts(infernix::text::GrammarCompiler& compiled) {
    const auto parameters = schema({{"type", "integer"}, {"minimum", 2}, {"maximum", 3}});
    std::vector<std::string> definitions;
    for (const char* name : {"strict", "loose", "excluded"}) {
        definitions.push_back(
            Json{{"type", "function"},
                 {"function",
                  {{"name", name},
                   {"parameters",
                    std::string_view(name) == "excluded" ? Json{{"type", "object"}} : parameters},
                   {"strict", std::string_view(name) != "loose"}}}}
                .dump());
    }
    const auto declarations = frontend::build_tool_call_output_contract(definitions);
    infernix::ToolChoice choice;
    choice.constraints   = infernix::ToolConstraintMode::Automatic;
    choice.allowed_names = std::vector<std::string>{"strict", "loose"};
    const auto selected  = frontend::select_tool_call_contract(declarations, choice);
    require(accepts(compiled, *selected, "ordinary answer"), "selection forced an auto call");
    require(!accepts(compiled, *selected, call("wrong", "strict")),
            "non-strict sibling relaxed strict values");
    require(!accepts(compiled, *selected, call("2", "excluded")),
            "selection admitted an excluded function");
    const auto mixed = call("2", "strict") + "\n" + call("wrong", "loose");
    require(accepts(compiled, *selected, mixed), "mixed strict/non-strict calls rejected");
    frontend::ToolCallOutputDecoder decoder(selected, 64);
    (void)feed(decoder, mixed);
    const auto calls = decoder.finish().tool_calls;
    require(calls.size() == 2 && Json::parse(calls[0].arguments_json)["x"] == 2 &&
                Json::parse(calls[1].arguments_json)["x"] == "wrong",
            "selected tools used another function's argument codec");

    choice.allowed_names = std::vector<std::string>{"loose"};
    choice.mode          = infernix::ToolChoiceMode::Required;
    choice.parallel      = false;
    const auto single    = frontend::select_tool_call_contract(declarations, choice);
    require(accepts(compiled, *single, call("wrong", "loose")) &&
                !accepts(compiled, *single, call("2", "strict")) &&
                !accepts(compiled, *single, "ordinary answer"),
            "reselecting declarations lost the required function policy");
    for (const auto& names : {std::vector<std::string>{}, std::vector<std::string>{"missing"}}) {
        choice.allowed_names = names;
        bool rejected        = false;
        try {
            (void)frontend::select_tool_call_contract(declarations, choice);
        } catch (const infernix::RequestError&) { rejected = true; }
        require(rejected, "required selection accepted an empty or unknown function set");
    }
}

void run() {
    auto compiled = compiler();
    basic_contracts(compiled);
    selected_contracts(compiled);
    infernix::ToolChoice required;
    required.mode = infernix::ToolChoiceMode::Required;
    auto fixed =
        contract(schema({{"type", "integer"}, {"minimum", 2}, {"maximum", 3}}), true, required);
    require(accepts(compiled, *fixed, call("2")), "strict integer rejected");
    require(!accepts(compiled, *fixed, call("4")), "integer range ignored");
    require(!accepts(compiled, *fixed, call("\"2\"")), "integer coerced from string");
    require(!accepts(compiled, *fixed, "ordinary answer"), "required choice admitted text");
    require(accepts(compiled, *fixed, call("2") + "\n" + call("3")), "parallel calls rejected");
    required.parallel = false;
    auto single       = contract(schema({{"type", "integer"}}), true, required);
    require(!accepts(compiled, *single, call("2") + "\n" + call("3")),
            "single choice admitted second call");
    auto automatic = contract(schema({{"type", "string"}}), true);
    require(accepts(compiled, *automatic, "ordinary answer"), "auto rejected plain text");
    require(accepts(compiled, *automatic, "Calling.\n\n" + call("你好")),
            "auto rejected preface/call");
    require(!accepts(compiled, *automatic,
                     "<tool_call>\n<function=unknown>\n</function>\n</tool_call>"),
            "auto escaped through free text");
    // Only the `<tool_call>` token opens a call; its spelling in ordinary tokens is free text that
    // the grammar admits and the decoder publishes as content.
    require(accepts_tokens(compiled, *automatic,
                           tokens("<tool_call>\n<function=unknown>\n</function>\n</tool_call>", true)),
            "auto read a spelled marker as a call opener");
    require(!accepts_tokens(compiled, *fixed, tokens(call("2"), true)),
            "a required call accepted a spelled opener");
    {
        const std::string text = "Calling.\n" + call("2");
        frontend::ToolCallOutputDecoder spelled(automatic, 64);
        std::string visible = spelled.feed({text, {}});
        const auto terminal = spelled.finish();
        require(visible + terminal.content == text && terminal.tool_calls.empty() &&
                    !terminal.diagnostics.marker_seen,
                "a spelled marker opened a constrained call");
    }
    auto disabled = required;
    disabled.mode = infernix::ToolChoiceMode::None;
    auto no_tools = contract(schema({{"type", "string"}}), true, disabled);
    require(accepts(compiled, *no_tools, "ordinary answer") &&
                !accepts(compiled, *no_tools, call("a")),
            "none allowed a tool marker");
    require(accepts_tokens(compiled, *no_tools, tokens(call("a"), true)),
            "none refused the marker spelled as text");
    infernix::ToolChoice basic;
    basic.constraints = infernix::ToolConstraintMode::Basic;
    auto structure    = contract(schema({{"type", "integer"}, {"minimum", 2}}), false, basic);
    require(accepts(compiled, *structure, call("not an integer")),
            "basic unexpectedly enforced argument types");
    auto bounded =
        contract(schema({{"type", "string"}, {"minLength", 1}, {"maxLength", 2}}), true, required);
    require(accepts(compiled, *bounded, call("😀")), "Unicode length mismatch");
    require(!accepts(compiled, *bounded, call("abc")), "length bound dropped");
    auto patterned = contract(schema({{"type", "string"}, {"pattern", "p"}}), true, required);
    require(accepts(compiled, *patterned, call("apple")), "raw pattern lost search semantics");
    require(!accepts(compiled, *patterned, call("none")), "raw pattern ignored");
    auto intersection = contract(
        schema({{"type", "string"}, {"pattern", "你|😀"}, {"minLength", 2}, {"maxLength", 3}}),
        true, required);
    require(accepts(compiled, *intersection, call("你好")) &&
                accepts(compiled, *intersection, call("😀ab")) &&
                !accepts(compiled, *intersection, call("你")) &&
                !accepts(compiled, *intersection, call("你abc")) &&
                !accepts(compiled, *intersection, call("abc")),
            "strict string conjunction lost pattern or Unicode length");
    auto conjunction = contract(schema({{"allOf",
                                         {Json{{"type", "string"}, {"pattern", "a"}},
                                          Json{{"pattern", "b"}, {"maxLength", 3}}}}}),
                                true, required);
    require(accepts(compiled, *conjunction, call("ab")) &&
                !accepts(compiled, *conjunction, call("aaaa")),
            "strict parameter domain disagrees with normalized conjunction");
    const auto value = std::string("  你好\n");
    frontend::ToolCallOutputDecoder decoder(automatic, 64);
    const auto text = call(value);
    for (std::size_t at = 0; at < text.size(); ++at)
        require(feed(decoder, text, at, 1).empty(), "tool bytes leaked");
    auto result = decoder.finish();
    require(result.tool_calls.size() == 1 &&
                Json::parse(result.tool_calls[0].arguments_json)["x"] == value,
            "raw string changed");
    auto nested           = contract(schema({{"type", "object"},
                                             {"properties", {{"s", {{"type", "string"}}}}},
                                             {"required", {"s"}},
                                             {"additionalProperties", false}}),
                                     true, required);
    const auto json_value = std::string(R"({"s": "</parameter>"})");
    require(accepts(compiled, *nested, call(json_value)), "nested marker rejected");
    frontend::ToolCallOutputDecoder json_decoder(nested, 64);
    (void)feed(json_decoder, call(json_value));
    require(Json::parse(json_decoder.finish().tool_calls[0].arguments_json)["x"]["s"] ==
                "</parameter>",
            "nested marker broke parser");
    const auto two = call("2") + "\n" + call("3");
    for (auto reason : {infernix::FinishReason::OutputLimit, infernix::FinishReason::Cancelled})
        for (std::size_t cut = 0; cut < two.size(); ++cut) {
            frontend::ToolCallOutputDecoder truncated(fixed, 64);
            (void)feed(truncated, std::string_view(two).substr(0, cut));
            const auto output = truncated.finish(reason);
            require(output.tool_calls.size() == (cut >= call("2").size() ? 1 : 0),
                    "interruption lost a complete call");
        }
    const auto seed = call("你好").substr(0, call("你好").size() - 20);
    frontend::ToolCallOutputDecoder continuation(automatic, 64);
    continuation.initialize_continuation(seed);
    auto match = frontend::compile_tool_grammar(compiled, *automatic, kToolCall, {}, tokens(seed));
    (void)feed(continuation, call("你好"), seed.size());
    require(continuation.finish().tool_calls.size() == 1, "continuation did not complete the call");
    // A nullable string (OpenAI strict mode's optional field: a type list of string and null, or an anyOf /
    // oneOf of strings and null) is raw text, as a string is; the text `null` reads as JSON null (BX-06).
    for (const Json& nullable :
         {Json{{"type", {"string", "null"}}},
          Json{{"anyOf", {{{"type", "string"}, {"maxLength", 100}}, {{"type", "null"}}}}},
          Json{{"oneOf", {{{"type", "null"}}, {{"type", "string"}}}}}}) {
        const auto bound = contract(schema(nullable), true, required);
        require(accepts(compiled, *bound, call("src/main.cpp")) && accepts(compiled, *bound, call("null")),
                ("a nullable string refused raw text or null: " + nullable.dump()).c_str());
        frontend::ToolCallOutputDecoder text_decoder(bound, 64);
        (void)feed(text_decoder, call("src/main.cpp"));
        require(Json::parse(text_decoder.finish().tool_calls.at(0).arguments_json)["x"] == "src/main.cpp",
                ("a nullable string's text changed: " + nullable.dump()).c_str());
        frontend::ToolCallOutputDecoder null_decoder(bound, 64);
        (void)feed(null_decoder, call("null"));
        require(Json::parse(null_decoder.finish().tool_calls.at(0).arguments_json)["x"].is_null(),
                ("a nullable string's null was not null: " + nullable.dump()).c_str());
    }
    // A string mixed with another JSON value type stays ambiguous.
    bool rejected = false;
    try {
        (void)contract(schema({{"anyOf", {{{"type", "string"}}, {{"type", "integer"}}}}}), true, required);
    } catch (const infernix::RequestError&) { rejected = true; }
    require(rejected, "ambiguous string and integer representation accepted");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            auto compiled = compiler();
            std::string line;
            while (std::getline(std::cin, line)) {
                Json response;
                try {
                    const auto input = Json::parse(line);
                    infernix::ToolChoice choice;
                    choice.mode           = infernix::ToolChoiceMode::Required;
                    auto bound            = contract(input.at("schema"), true, choice);
                    response["accepted"]  = Json::array();
                    response["arguments"] = Json::array();
                    for (const auto& text : input.at("candidates")) {
                        bool allowed = accepts(compiled, *bound, text.get<std::string>());
                        response["accepted"].push_back(allowed);
                        if (allowed) {
                            frontend::ToolCallOutputDecoder decode(bound, 64);
                            (void)feed(decode, text.get<std::string>());
                            response["arguments"].push_back(
                                Json::parse(decode.finish().tool_calls[0].arguments_json));
                        } else
                            response["arguments"].push_back(nullptr);
                    }
                } catch (const std::exception& error) { response = {{"error", error.what()}}; }
                std::cout << response.dump() << '\n';
            }
        } else {
            run();
            std::cout << "tool constraints passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
