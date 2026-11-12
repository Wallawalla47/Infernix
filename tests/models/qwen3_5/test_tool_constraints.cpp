#include "models/qwen3_5/frontend/tool_contract.h"
#include "models/qwen3_5/frontend/tool_grammar.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/grammar.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
namespace frontend = infernix::models::qwen3_5::frontend;
using Json         = nlohmann::ordered_json;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Byte tokens 0-255, EOS 256, the model's `<tool_call>` token and its `</think>` token.
constexpr std::int32_t kEos      = 256;
constexpr std::int32_t kToolCall = 257;
constexpr std::int32_t kThinkEnd = 258;

infernix::text::GrammarCompiler compiler() {
    std::vector<std::string> tokens;
    for (int i = 0; i < 256; ++i) tokens.emplace_back(1, static_cast<char>(i));
    tokens.emplace_back();
    tokens.emplace_back(frontend::kToolCallOpen);
    tokens.emplace_back("</think>");
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
        // The engine reports where a token starts even when the cut falls inside its bytes.
        (void)feed(decoder, two, 0, cut);
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

// A `<tool_call>` token inside Markdown code of an auto answer is a quoted example on the constrained
// route as on the free one (Strata measured 9 of 9 such examples run as calls before): the grammar
// stays in the answer's text there, so the answer goes on, and the decoder publishes the token's
// spelling as content, at every chunking. A call outside code still opens, and a lookahead through
// a quoted token finds no dead end.
void quoted_calls(infernix::text::GrammarCompiler& compiled) {
    // A call quoted inside a block quote's fence: every line after the quote's `> `.
    std::string in_quote = "> " + call("a");
    for (auto at = in_quote.find('\n'); at != std::string::npos; at = in_quote.find('\n', at + 1)) {
        in_quote.insert(at + 1, "> ");
    }
    const std::vector<std::string> quoted{"Format:\n```xml\n" + call("a") + "\n```\nDone.",
                                          "````markdown\n```xml\n" + call("a") + "\n```\n````\nDone.",
                                          "Use `" + call("a") + "` like this.",
                                          "Example:\n\n    " + call("a") + "\n\nDone.",
                                          "> Format:\n> ```xml\n" + in_quote + "\n> ```\nDone."};
    // A span left open by a stray backtick on an earlier line does not reach the call at a line's start
    // (Strata #4923); a list item's continuation paragraph four columns in is the item's text, not code.
    const std::vector<std::string> real{"```\ncode\n```\n" + call("a"), "Example:\n    " + call("a"),
                                        "Use `x` then " + call("a"), "Let me use `run_commands to check.\n" + call("a"),
                                        "- Step one.\n\n    " + call("a")};
    for (const bool strict : {false, true}) {
        const auto bound = contract(schema({{"type", "string"}}), strict);
        for (const std::string& text : quoted) {
            require(accepts(compiled, *bound, text), "the constraint opened a call at a token quoted in Markdown code");
            for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{7}, text.size()}) {
                frontend::ToolCallOutputDecoder decoder(bound, 64);
                std::string content;
                for (std::size_t at = 0; at < text.size(); at += width) { content += feed(decoder, text, at, width); }
                const auto terminal = decoder.finish();
                require(terminal.tool_calls.empty() && content + terminal.content == text,
                        "the decoder read a quoted call token as a call or lost bytes");
            }
        }
        for (const std::string& text : real) {
            require(accepts(compiled, *bound, text), "the constraint refused a call outside Markdown code");
            frontend::ToolCallOutputDecoder decoder(bound, 64);
            (void)feed(decoder, text);
            require(decoder.finish().tool_calls.size() == 1, "a call outside Markdown code did not open");
        }
        require(!accepts(compiled, *bound, "Use it: " + std::string(frontend::kToolCallOpen) + "` here."),
                "a call token outside Markdown code did not open a call");
        // Drafts through a quoted token: after it the answer's text goes on (inside a call the
        // backtick that follows would be refused).
        auto matcher      = frontend::compile_tool_grammar(compiled, *bound, kToolCall, {}, {});
        const auto drafts = tokens("Use `" + std::string(frontend::kToolCallOpen) + "` here.");
        const auto after  = static_cast<std::size_t>(std::find(drafts.begin(), drafts.end(), kToolCall) - drafts.begin()) + 1;
        const std::size_t stride = matcher->mask_words();
        std::vector<std::uint32_t> words(stride * (drafts.size() + 1));
        require(matcher->masks(drafts, words) == 0, "a lookahead through a quoted call token found a dead end");
        for (std::size_t p = after; p < drafts.size(); ++p) {
            const auto token = static_cast<std::size_t>(drafts[p]);
            require((words[p * stride + token / 32] >> (token % 32) & 1U) != 0,
                    "a lookahead refused the answer's text after a quoted call token");
        }
    }
}

// A raw value holds the whole call's closing lines when more value text follows (a file that documents
// the call format, Strata #4701, where the value was cut there): the grammar keeps the value open
// after them, the call still ends at its own closers before the end or the next call's token, and the
// decoder publishes the whole value, its streamed steps concatenating to the call. Text after closers
// with no later closer cannot end the output, and a bounded string keeps the plain exclusion.
void quoted_closers(infernix::text::GrammarCompiler& compiled) {
    const std::string closers  = "\n</parameter>\n</function>\n</tool_call>";
    const std::string document = "A call ends with:" + closers + "\nThen the reply goes on.";
    for (const bool strict : {false, true}) {
        const auto bound = contract(schema({{"type", "string"}}), strict);
        require(accepts(compiled, *bound, call(document)), "the grammar cut a value at quoted closing lines");
        require(accepts(compiled, *bound, "Writing.\n" + call(document) + "\n" + call("b")),
                "a value with quoted closing lines and a following call were refused");
        require(accepts(compiled, *bound, call("a")), "a plain call no longer ends at its closers");
        require(!accepts(compiled, *bound, call("a") + "\nThat wrote it."), "a call ended with prose after it");
        for (const std::size_t width : {std::size_t{1}, std::size_t{5}, std::size_t{4096}}) {
            const std::string text = "Writing.\n" + call(document);
            frontend::ToolCallOutputDecoder decoder(bound, 64);
            std::string streamed;
            const auto take = [&] {
                for (const infernix::ToolCallDelta& delta : decoder.take_call_deltas()) { streamed += delta.arguments; }
            };
            for (std::size_t at = 0; at < text.size(); at += width) {
                (void)feed(decoder, text, at, width);
                take();
            }
            const auto terminal = decoder.finish();
            take();
            require(terminal.tool_calls.size() == 1 &&
                        Json::parse(terminal.tool_calls[0].arguments_json) == Json{{"x", document}} &&
                        streamed == terminal.tool_calls[0].arguments_json,
                    "the decoder cut a value at quoted closing lines or streamed it differently");
        }
        // Drafts through the closers: the text after them is admitted, and so is the end after the
        // value's own closers.
        auto matcher = frontend::compile_tool_grammar(compiled, *bound, kToolCall, {}, {});
        for (const std::int32_t token : tokens("<tool_call>\n<function=call>\n<parameter=x>\na" + closers)) {
            matcher->accept(token);
        }
        matcher->confirm();
        const auto drafts        = tokens("\nb" + closers);
        const std::size_t stride = matcher->mask_words();
        std::vector<std::uint32_t> words(stride * (drafts.size() + 1));
        require(matcher->masks(drafts, words) == 0, "a lookahead through quoted closing lines found a dead end");
        for (std::size_t p = 0; p < drafts.size(); ++p) {
            const auto token = static_cast<std::size_t>(drafts[p]);
            require((words[p * stride + token / 32] >> (token % 32) & 1U) != 0,
                    "a lookahead refused the value text after quoted closing lines");
        }
        require((words[drafts.size() * stride + kEos / 32] >> (kEos % 32) & 1U) != 0,
                "a value's own closers after quoted ones did not admit the end");
    }
    const auto bounded = contract(schema({{"type", "string"}, {"maxLength", 100}}), true);
    require(!accepts(compiled, *bounded, call(document)), "a bounded string held quoted closing lines");
}

// Inside the reasoning a tool grammar refuses the `<tool_call>` token, which could never open a call
// there and would leave the model unable to end the turn (EOS is refused until the reasoning closes;
// ninferpro's incident: 65 and 145 calls written inside thinking). Its spelling in ordinary tokens stays
// reasoning text; after `</think>` and the separator the token opens calls as before. An assistant
// continuation holding the token inside the reasoning is not a prefix of the grammar.
void calls_in_thinking(infernix::text::GrammarCompiler& compiled) {
    const infernix::text::ReasoningFraming thinking{.close_token = kThinkEnd, .separator = "\n\n"};
    infernix::ToolChoice required;
    required.mode = infernix::ToolChoiceMode::Required;
    const auto allowed = [](const std::vector<std::uint32_t>& words, std::int32_t token) {
        return (words[static_cast<std::size_t>(token) / 32] >> (token % 32) & 1U) != 0;
    };
    for (const bool strict : {false, true}) {
        for (const auto& choice : {infernix::ToolChoice{}, required}) {
            const auto bound = contract(schema({{"type", "string"}}), strict, choice);
            auto matcher     = frontend::compile_tool_grammar(compiled, *bound, kToolCall, thinking, {});
            std::vector<std::uint32_t> words(matcher->mask_words());
            const auto step = [&](std::int32_t token) {
                require(matcher->masks({}, words) == 0 && allowed(words, token),
                        "the grammar refused a token of the reasoning or of the call after it");
                matcher->accept(token);
                matcher->confirm();
            };
            for (const std::int32_t token : tokens("Let me check the file. ")) { step(token); }
            require(matcher->masks({}, words) == 0 && !allowed(words, kToolCall) && !allowed(words, kEos) &&
                        allowed(words, kThinkEnd),
                    "the reasoning admitted the call token or the end, or refused its close");
            for (const std::int32_t token : tokens("```\n" + call("a"), true)) { step(token); }
            require(matcher->masks({}, words) == 0 && !allowed(words, kToolCall) && !allowed(words, kEos),
                    "a call spelled in ordinary tokens left the reasoning");
            step(kThinkEnd);
            for (const std::int32_t token : tokens("\n\n")) { step(token); }
            for (const std::int32_t token : tokens(call("a"))) { step(token); }
            require(matcher->masks({}, words) == 0 && allowed(words, kEos),
                    "a call after the reasoning did not complete the turn");
        }
    }
    const auto automatic = contract(schema({{"type", "string"}}), false);
    bool refused         = false;
    try {
        std::vector<std::int32_t> continuation = tokens("Plan: ");
        continuation.push_back(kToolCall);
        (void)frontend::compile_tool_grammar(compiled, *automatic, kToolCall, thinking, continuation);
    } catch (const infernix::RequestError&) { refused = true; }
    require(refused, "a continuation with the call token inside the reasoning was accepted");
}

// The grammar's raw string reads the `<tool_call>` token right after a call's closing lines (or after
// them and their `\n` when calls are not parallel) as value text, which the decoder refuses as a
// malformed call: the token is masked there for every choice, and a newline still opens the next call.
void calls_after_closers(infernix::text::GrammarCompiler& compiled) {
    infernix::ToolChoice required;
    required.mode = infernix::ToolChoiceMode::Required;
    infernix::ToolChoice single;
    single.parallel = false;
    for (const bool strict : {false, true}) {
        for (const auto& choice : {infernix::ToolChoice{}, required}) {
            const auto bound = contract(schema({{"type", "string"}}), strict, choice);
            require(!accepts(compiled, *bound, call("a") + call("b")),
                    "the call token right after a call's closing lines was admitted");
            require(accepts(compiled, *bound, call("a") + "\n" + call("b")),
                    "a call after a call's closing lines and a newline was refused");
        }
        const auto one = contract(schema({{"type", "string"}}), strict, single);
        require(accepts(compiled, *one, call("a")), "a single call was refused");
        require(!accepts(compiled, *one, call("a") + "\n" + call("b")),
                "a second call token was admitted where calls are not parallel");
    }
    // Its decoder still refuses that sequence, so the mask is what keeps the request alive.
    const auto bound = contract(schema({{"type", "string"}}), false);
    frontend::ToolCallOutputDecoder decoder(bound, 64);
    (void)feed(decoder, call("a") + call("b"));
    bool refused = false;
    try {
        (void)decoder.finish();
    } catch (const std::logic_error&) { refused = true; }
    require(refused, "the decoder accepted a call token right after a call's closing lines");
}

// A string value that never closes (Strata #5756, strata-nvfp4 #24) on the constrained route. A parameter
// name holds no line break, so an opener that lost its `>` cannot be generated (the grammar admits `>`
// there). A raw value ends only at `\n</parameter>`: after an inline nested pair, or after a closer that
// lost its `>`, the call's own closers are value text and the turn cannot end inside the value; a later
// closer ends it, and the decoder publishes the value the grammar read.
void unclosed_values(infernix::text::GrammarCompiler& compiled) {
    const auto allowed = [](const std::vector<std::uint32_t>& words, std::int32_t token) {
        return (words[static_cast<std::size_t>(token) / 32] >> (token % 32) & 1U) != 0;
    };
    const Json properties{{"path", {{"type", "string"}}}, {"new_string", {{"type", "string"}}}};
    const std::string path = "<tool_call>\n<function=call>\n<parameter=path>\ndocs/tools.md\n</parameter>\n";
    for (const bool strict : {false, true}) {
        Json source{{"type", "object"}, {"properties", properties}};
        if (strict) {
            source["required"]             = Json::array({"path", "new_string"});
            source["additionalProperties"] = false;
        }
        const auto bound = contract(source, strict);
        // The mask after `text`, every token of which the grammar admits.
        const auto mask_after = [&](const std::string& text) {
            auto matcher = frontend::compile_tool_grammar(compiled, *bound, kToolCall, {}, {});
            std::vector<std::uint32_t> words(matcher->mask_words());
            for (const std::int32_t token : tokens(text)) {
                require(matcher->masks({}, words) == 0 && allowed(words, token),
                        "the grammar refused a token of an unclosed value's prefix");
                matcher->accept(token);
                matcher->confirm();
            }
            require(matcher->masks({}, words) == 0, "the grammar found a dead end in an unclosed value");
            return words;
        };
        for (const std::string& opener : {std::string("<tool_call>\n<function=call>\n<parameter=path"),
                                          path + "<parameter=new_string"}) {
            const auto words = mask_after(opener);
            require(!allowed(words, '\n') && allowed(words, '>'), "a parameter name admitted a line break");
        }
        for (const std::string value : {std::string("<parameter=limit>25</parameter>"), std::string("hello\n</parameter")}) {
            const std::string open = path + "<parameter=new_string>\n" + value + "\n</function>\n</tool_call>";
            require(!allowed(mask_after(open), kEos), "the turn could end inside an unclosed value");
            const std::string closed = open + "\n</parameter>\n</function>\n</tool_call>";
            require(accepts(compiled, *bound, closed), "a later closer did not end an unclosed value");
            frontend::ToolCallOutputDecoder decoder(bound, 64);
            (void)feed(decoder, closed);
            const auto terminal = decoder.finish();
            require(terminal.tool_calls.size() == 1 &&
                        Json::parse(terminal.tool_calls[0].arguments_json) ==
                            Json{{"path", "docs/tools.md"}, {"new_string", value + "\n</function>\n</tool_call>"}},
                    "the decoder did not publish the value the grammar read");
        }
    }
}

// A parameter a declared non-strict tool does not declare counts as a schema mismatch on the constrained
// (streamed) route, as the whole-text route counts it (candidates QA-14).
void undeclared_parameters() {
    const auto open        = contract(schema({{"type", "string"}}), false);
    const std::string text = "<tool_call>\n<function=call>\n<parameter=extra>\n1\n</parameter>\n</function>\n</tool_call>";
    frontend::ToolCallOutputDecoder decoder(open, 64);
    (void)feed(decoder, text);
    const auto terminal = decoder.finish();
    require(terminal.tool_calls.size() == 1 && terminal.diagnostics.schema_mismatch_arguments == 1,
            "a streamed call's undeclared parameter was not counted as a schema mismatch");
}

void run() {
    auto compiled = compiler();
    undeclared_parameters();
    basic_contracts(compiled);
    selected_contracts(compiled);
    quoted_calls(compiled);
    quoted_closers(compiled);
    calls_in_thinking(compiled);
    calls_after_closers(compiled);
    unclosed_values(compiled);
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
    // A nullable string's enum or const keeps its members (NInfer #425's status filter, its anyOf
    // form, Pydantic's Optional[Status] through a reference): the grammar takes the members and the
    // text `null`, which reads as JSON null, and refuses other text. Bounds or a pattern that refuse
    // the four characters still leave `null` to the grammar.
    const auto decodes = [](const auto& bound, std::string_view text) {
        frontend::ToolCallOutputDecoder decoder(bound, 64);
        (void)feed(decoder, call(text));
        return Json::parse(decoder.finish().tool_calls.at(0).arguments_json)["x"];
    };
    const Json statuses = Json::array({"pending", "completed"});
    Json referenced = schema({{"anyOf", {{{"$ref", "#/$defs/Status"}}, {{"type", "null"}}}}});
    referenced["$defs"] = {{"Status", {{"type", "string"}, {"enum", statuses}}}};
    for (const Json& parameters :
         {schema({{"type", {"string", "null"}}, {"enum", {"pending", "completed", nullptr}}}),
          schema({{"anyOf", {{{"type", "string"}, {"enum", statuses}}, {{"type", "null"}}}}}),
          referenced}) {
        const auto bound = contract(parameters, true, required);
        require(accepts(compiled, *bound, call("pending")) &&
                    accepts(compiled, *bound, call("completed")) &&
                    accepts(compiled, *bound, call("null")) &&
                    !accepts(compiled, *bound, call("other")) &&
                    !accepts(compiled, *bound, call("Pending")),
                ("a nullable enum refused a member or null, or admitted other text: " +
                 parameters.dump())
                    .c_str());
        require(decodes(bound, "pending") == "pending" && decodes(bound, "null").is_null(),
                ("a nullable enum's member or null decoded wrongly: " + parameters.dump()).c_str());
    }
    for (const auto& [parameters, text, refused] :
         {std::tuple{schema({{"type", {"string", "null"}}, {"maxLength", 2}}), "ab", "abc"},
          std::tuple{schema({{"type", {"string", "null"}}, {"pattern", "^[0-9]+$"}}), "12", "ab"}}) {
        const auto bound = contract(parameters, true, required);
        require(accepts(compiled, *bound, call(text)) && accepts(compiled, *bound, call("null")) &&
                    !accepts(compiled, *bound, call(refused)),
                ("a bounded nullable string refused null: " + parameters.dump()).c_str());
        require(decodes(bound, "null").is_null() && decodes(bound, text) == text,
                ("a bounded nullable string decoded wrongly: " + parameters.dump()).c_str());
    }
    // The string "null" beside null has no raw text of its own.
    for (const Json& ambiguous :
         {schema({{"type", {"string", "null"}}, {"enum", {"null", "a", nullptr}}}),
          schema({{"anyOf", {{{"const", "null"}}, {{"type", "null"}}}}})}) {
        bool refused = false;
        try {
            (void)contract(ambiguous, true, required);
        } catch (const infernix::RequestError&) { refused = true; }
        require(refused, ("the string \"null\" beside null was accepted: " + ambiguous.dump()).c_str());
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
