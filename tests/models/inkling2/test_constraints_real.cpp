// Inkling 2's constrained route (src/models/inkling2/frontend/constraints.h) through the frontend and the output
// session on the real tokenizer and the official template (CPU only):
// 1. Walks: for each constrained turn (every content kind, the tool policies, a JSON answer beside tools, a thinking
//    budget), seeded walks draw each token from its position's mask (the stop token, the markers and one-byte tokens
//    favoured so that walks end), preview it and commit it. The session never refuses a legal token, and a walk that
//    ends at the stop token publishes what the language says: the JSON answer parses with the schema's shape, the
//    regex and choice texts match, the calls name callable tools with object arguments (a strict tool's with its
//    schema's shape), a required choice calls, none never does, parallel false calls once.
// 2. The framing: a walk's first mask admits only <|message_model|>; a token outside a mask is refused (the request's
//    error) and the session goes on from its committed state; a thinking budget's control is accepted.
// 3. Refusals: custom stops, raw output, a regex answer beside tools, a named choice of an undeclared tool.
// 4. Mask cost, printed (not asserted): microseconds a mask by its legal-token count (free text, a JSON string,
//    structure), and the first mask's wait for the compile beside the prefill.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR  tokenizer.json, tokenizer_config.json and generation_config.json

#include "models/inkling2/frontend/frontend.h"
#include "runtime/contract/execution.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::json;
namespace ink = infernix::models::inkling2;
using infernix::FinishReason;
using infernix::OutputChannel;
using infernix::OutputConstraint;
using infernix::TokenId;
using infernix::ToolChoice;
using infernix::ToolChoiceMode;
using infernix::ToolConstraintMode;
using Clock = std::chrono::steady_clock;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

bool throws(const std::function<void()>& body) {
    try {
        body();
    } catch (...) { return true; }
    return false;
}

// Mask timings by the mask's legal-token count.
struct Costs {
    std::map<std::string, std::vector<double>> us;
    std::vector<double> first_wait_us;

    void add(std::size_t legal, double micros) {
        const char* kind = legal > 150000 ? "free text (>150K legal)" : legal > 1000 ? "string-like (1K-150K)" : "structure (<=1K)";
        us[kind].push_back(micros);
    }
};

struct Walk {
    std::string reasoning, content;
    std::vector<infernix::GeneratedToolCall> calls;
    FinishReason finish = FinishReason::None;
    std::size_t steps   = 0;
    bool applied        = false; // a thinking budget's control was applied
};

struct Context {
    const ink::Frontend& frontend;
    const infernix::text::Tokenizer& tokenizer;
    std::vector<TokenId> markers; // the seven, then the stop token
    TokenId stop = -1;
    TokenId message_model = -1;
    std::size_t words = 0;
    Costs costs;
};

std::vector<TokenId> legal_tokens(const std::vector<std::uint32_t>& mask) {
    std::vector<TokenId> out;
    for (std::size_t w = 0; w < mask.size(); ++w) {
        for (std::uint32_t bits = mask[w]; bits != 0; bits &= bits - 1) {
            out.push_back(static_cast<TokenId>(w * 32 + static_cast<std::size_t>(std::countr_zero(bits))));
        }
    }
    return out;
}

bool allowed(const std::vector<std::uint32_t>& mask, TokenId token) {
    return (mask[static_cast<std::size_t>(token) / 32] >> (static_cast<std::size_t>(token) % 32) & 1U) != 0;
}

// One walk to the stop token or `limit` tokens.
Walk walk(Context& c, ink::OutputSession& session, std::mt19937& rng, std::size_t limit, const std::string& name,
          bool may_end_empty) {
    Walk out;
    std::vector<std::uint32_t> mask(c.words);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t step = 0; step < limit; ++step) {
        const bool first  = step == 0;
        const auto before = Clock::now();
        const std::uint64_t dead = session.grammar_masks({}, mask);
        const double micros      = std::chrono::duration<double, std::micro>(Clock::now() - before).count();
        check(dead == 0, name + ": a reachable position has no legal token");
        if (dead != 0) { break; }
        const std::vector<TokenId> legal = legal_tokens(mask);
        first ? c.costs.first_wait_us.push_back(micros) : c.costs.add(legal.size(), micros);
        if (first) {
            // A turn opens a block; a tool turn that may answer in text may also end empty, as a free turn may.
            const bool block = legal == std::vector<TokenId>{c.message_model};
            const bool empty = legal == std::vector<TokenId>{c.message_model, c.stop} ||
                               legal == std::vector<TokenId>{c.stop, c.message_model};
            check(block || (may_end_empty && empty), name + ": the first mask admits <|message_model|> (and the stop token "
                                                            "where the turn may end empty) only");
        }
        std::vector<TokenId> special, single;
        for (const TokenId t : legal) {
            if (std::find(c.markers.begin(), c.markers.end(), t) != c.markers.end()) {
                special.push_back(t);
            } else if (c.tokenizer.decoded_token(t).bytes.size() == 1) {
                single.push_back(t);
            }
        }
        const double u = unit(rng);
        const std::vector<TokenId>& pool = u < 0.35 && !special.empty() ? special : u < 0.75 && !single.empty() ? single : legal;
        const TokenId token = pool[std::uniform_int_distribution<std::size_t>(0, pool.size() - 1)(rng)];
        // A token outside the mask is refused, and the session goes on from its committed state.
        if (step == 3) {
            TokenId outside = -1;
            for (const TokenId candidate : {c.stop, c.message_model, c.markers[6], TokenId{0}, TokenId{17}}) {
                if (!allowed(mask, candidate)) {
                    outside = candidate;
                    break;
                }
            }
            if (outside >= 0) {
                bool refused = false;
                try {
                    (void)session.preview_model({&outside, 1}, static_cast<std::uint32_t>(limit - step), FinishReason::OutputLimit);
                } catch (const infernix::runtime::RequestOutputError&) { refused = true; }
                check(refused, name + ": a token outside the mask is refused");
                session.discard_preview();
            }
        }
        const auto decision =
            session.preview_model({&token, 1}, static_cast<std::uint32_t>(limit - step), FinishReason::OutputLimit);
        ink::PublishedOutput published = session.commit_preview();
        for (const auto& delta : published) {
            (delta.channel == OutputChannel::Reasoning ? out.reasoning : out.content) += delta.text;
        }
        ++out.steps;
        if (decision.continuation == infernix::runtime::ContinuationAction::ApplyTargetControl) {
            const std::vector<TokenId> control(session.pending_control_tokens().begin(), session.pending_control_tokens().end());
            check(!throws([&] {
                      (void)session.preview_control(control, static_cast<std::uint32_t>(limit - step - 1));
                      (void)session.commit_preview();
                  }),
                  name + ": the thinking control is accepted");
            out.applied = true;
            continue;
        }
        if (decision.finish_reason != FinishReason::None) {
            out.finish = decision.finish_reason;
            break;
        }
    }
    out.calls = session.take_tool_calls();
    return out;
}

infernix::PromptInput question(const std::vector<std::string>& tools = {}) {
    infernix::PromptInput input;
    input.messages.push_back({.role = infernix::ChatRole::User, .parts = {{.text = "Answer as asked."}}});
    input.options.tool_jsons = tools;
    return input;
}

const std::string kWeather =
    R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}},"required":["city"]}}})";
const std::string kSearchStrict =
    R"({"type":"function","function":{"name":"search","strict":true,"parameters":{"type":"object","properties":{"q":{"type":"string"},"limit":{"type":"integer"}},"required":["q","limit"],"additionalProperties":false}}})";

struct Case {
    std::string name;
    std::vector<std::string> tools;
    std::optional<OutputConstraint> constraint;
    ToolChoice choice;
    infernix::ThinkingControlOptions thinking;
    // Checks a walk that ended at the stop token.
    std::function<void(const Walk&, const std::string&)> ended;
};

// A JSON text's syntax, with the type of each top-level member: a walk's random numbers outgrow what a JSON library
// stores (`383E529`, a 20-digit integer), which is valid JSON and the grammar's to allow.
struct JsonShape {
    bool ok = false;
    std::string type;
    std::map<std::string, std::string> members;
};

class JsonSyntax {
public:
    explicit JsonSyntax(std::string_view text) : s_(text) {}

    JsonShape shape() {
        JsonShape out;
        out.type = value(&out.members);
        ws();
        out.ok   = !out.type.empty() && i_ == s_.size();
        return out;
    }

private:
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) { ++i_; }
    }
    bool digits() {
        const std::size_t start = i_;
        while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; }
        return i_ > start;
    }
    bool string(std::string* out) {
        if (i_ >= s_.size() || s_[i_] != '"') { return false; }
        for (++i_; i_ < s_.size(); ++i_) {
            const auto c = static_cast<unsigned char>(s_[i_]);
            if (c == '"') {
                ++i_;
                return true;
            }
            if (c < 0x20) { return false; }
            if (c == '\\') {
                if (++i_ >= s_.size()) { return false; }
                if (s_[i_] == 'u') {
                    for (int k = 0; k < 4; ++k) {
                        if (++i_ >= s_.size() || !std::isxdigit(static_cast<unsigned char>(s_[i_]))) { return false; }
                    }
                } else if (std::string_view("\"\\/bfnrt").find(s_[i_]) == std::string_view::npos) {
                    return false;
                }
            }
            if (out != nullptr) { out->push_back(static_cast<char>(c)); }
        }
        return false;
    }
    // The value's type ("" when it is not JSON); an object's members' types into `members` (top level only).
    std::string value(std::map<std::string, std::string>* members) {
        ws();
        if (i_ >= s_.size()) { return ""; }
        const char c = s_[i_];
        if (c == '{' || c == '[') {
            const bool object = c == '{';
            ++i_;
            ws();
            if (i_ < s_.size() && s_[i_] == (object ? '}' : ']')) {
                ++i_;
                return object ? "object" : "array";
            }
            for (;;) {
                std::string key;
                if (object) {
                    ws();
                    if (!string(&key)) { return ""; }
                    ws();
                    if (i_ >= s_.size() || s_[i_++] != ':') { return ""; }
                }
                const std::string type = value(nullptr);
                if (type.empty()) { return ""; }
                if (object && members != nullptr) { (*members)[key] = type; }
                ws();
                if (i_ >= s_.size()) { return ""; }
                if (s_[i_] == ',') {
                    ++i_;
                    continue;
                }
                if (s_[i_++] != (object ? '}' : ']')) { return ""; }
                return object ? "object" : "array";
            }
        }
        if (c == '"') { return string(nullptr) ? "string" : ""; }
        for (const std::string_view word : {"true", "false", "null"}) {
            if (s_.substr(i_, word.size()) == word) {
                i_ += word.size();
                return word == "null" ? "null" : "boolean";
            }
        }
        if (i_ < s_.size() && s_[i_] == '-') { ++i_; }
        if (i_ >= s_.size()) { return ""; }
        if (s_[i_] == '0') {
            ++i_;
        } else if (!digits()) {
            return "";
        }
        bool integer = true;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            integer = false;
            if (!digits()) { return ""; }
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            integer = false;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) { ++i_; }
            if (!digits()) { return ""; }
        }
        return integer ? "integer" : "number";
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

JsonShape json_shape(const std::string& text) { return JsonSyntax(text).shape(); }


} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_INKLING_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory), source(INFERNIX_SOURCE_DIR);
        const std::string tokenizer_json = read_file(root / "tokenizer.json");
        const std::string config         = read_file(root / "tokenizer_config.json");
        const std::string generation     = read_file(root / "generation_config.json");
        const std::string chat_template  = read_file(source / "tools" / "chat_templates" / "inkling2.jinja");
        auto tokenizer = std::make_shared<const infernix::text::Tokenizer>(infernix::text::TokenizerResources{
            .tokenizer_json         = tokenizer_json,
            .tokenizer_config_json  = config,
            .generation_config_json = generation,
            .pipeline               = infernix::text::TokenizerPipeline::Inkling});
        const ink::FrontendResources resources{.tokenizer_json         = tokenizer_json,
                                               .tokenizer_config_json  = config,
                                               .chat_template_jinja    = chat_template,
                                               .generation_config_json = generation,
                                               .tokenizer              = tokenizer};
        const ink::Frontend frontend(resources, {.max_context = 8192, .grammar_cache_bytes = 64ULL << 20});
        const auto warm = Clock::now();
        frontend.warm_constraints();
        std::cout << "vocabulary index built in "
                  << std::chrono::duration<double>(Clock::now() - warm).count() << " s\n";

        const auto id = [&](const char* spelling) { return static_cast<TokenId>(tokenizer->find_added_token(spelling)->id); };
        Context c{.frontend = frontend, .tokenizer = *tokenizer};
        c.message_model = id("<|message_model|>");
        c.stop          = id("<|content_model_end_sampling|>");
        c.markers       = {c.message_model,          id("<|content_thinking|>"),         id("<|content_text|>"),
                           id("<|content_tool_error|>"), id("<|content_invoke_tool_text|>"), id("<|content_invoke_tool_json|>"),
                           id("<|end_message|>"),    c.stop};
        c.words = (tokenizer->vocab_size() + 31) / 32;

        const auto basic = [](ToolChoiceMode mode) { return ToolChoice{.mode = mode, .constraints = ToolConstraintMode::Basic}; };
        const auto callable = [](const Walk& w, const std::string& name, const std::vector<std::string>& names) {
            for (const auto& call : w.calls) {
                const JsonShape args = json_shape(call.arguments_json);
                check(std::find(names.begin(), names.end(), call.name) != names.end(), name + ": a call names a callable tool");
                check(args.ok && args.type == "object", name + ": a call's arguments are an object: " + call.arguments_json);
            }
        };
        const std::string schema =
            R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false})";
        std::vector<Case> cases = {
            {"json schema", {}, OutputConstraint::json_schema(schema), {}, {},
             [&](const Walk& w, const std::string& n) {
                 const JsonShape shape = json_shape(w.content);
                 check(shape.ok && shape.type == "object" &&
                           shape.members == std::map<std::string, std::string>{{"name", "string"}, {"age", "integer"}},
                       n + ": the answer has the schema's shape: " + w.content);
             }},
            {"json object", {}, OutputConstraint::json_object(), {}, {},
             [&](const Walk& w, const std::string& n) {
                 const JsonShape shape = json_shape(w.content);
                 check(shape.ok && shape.type == "object", n + ": the answer is a JSON object: " + w.content);
             }},
            {"regex", {}, OutputConstraint::regex("[0-9]{3}-[a-z]+"), {}, {},
             [&](const Walk& w, const std::string& n) {
                 check(std::regex_match(w.content, std::regex("[0-9]{3}-[a-z]+")), n + ": the answer matches: " + w.content);
             }},
            {"choice", {}, OutputConstraint::choice({"red", "green", "blue"}), {}, {},
             [&](const Walk& w, const std::string& n) {
                 check(w.content == "red" || w.content == "green" || w.content == "blue", n + ": a choice: " + w.content);
             }},
            {"gbnf", {}, OutputConstraint::grammar(R"(root ::= "yes" | "no" [0-9]*)"), {}, {},
             [&](const Walk& w, const std::string& n) {
                 check(std::regex_match(w.content, std::regex("yes|no[0-9]*")), n + ": the grammar's text: " + w.content);
             }},
            {"gbnf free text", {}, OutputConstraint::grammar(R"(root ::= "x" ExcludeToken(17)*)"), {}, {},
             [&](const Walk& w, const std::string& n) { check(!w.content.empty() && w.content[0] == 'x', n + ": starts x"); }},
            {"thinking budget", {}, OutputConstraint::json_object(), {}, {.budget = 3},
             [&](const Walk& w, const std::string& n) {
                 const JsonShape shape = json_shape(w.content);
                 check(shape.ok && shape.type == "object", n + ": the answer is a JSON object: " + w.content);
             }},
            {"tools auto (basic)", {kWeather, kSearchStrict}, std::nullopt, basic(ToolChoiceMode::Auto), {},
             [&](const Walk& w, const std::string& n) { callable(w, n, {"get_weather", "search"}); }},
            {"tools required", {kWeather, kSearchStrict}, std::nullopt, basic(ToolChoiceMode::Required), {},
             [&](const Walk& w, const std::string& n) {
                 check(!w.calls.empty(), n + ": a required choice calls");
                 check(w.content.empty(), n + ": a required choice writes no text answer");
                 callable(w, n, {"get_weather", "search"});
                 for (const auto& call : w.calls) {
                     if (call.name == "search") {
                         check(json_shape(call.arguments_json).members ==
                                   std::map<std::string, std::string>{{"q", "string"}, {"limit", "integer"}},
                               n + ": a strict call's arguments have its schema's shape: " + call.arguments_json);
                     }
                 }
             }},
            {"tools named", {kWeather, kSearchStrict}, std::nullopt,
             ToolChoice{.mode = ToolChoiceMode::Required, .allowed_names = std::vector<std::string>{"search"}}, {},
             [&](const Walk& w, const std::string& n) {
                 check(!w.calls.empty(), n + ": a named choice calls");
                 callable(w, n, {"search"});
             }},
            {"tools none", {kWeather}, std::nullopt, basic(ToolChoiceMode::None), {},
             [&](const Walk& w, const std::string& n) { check(w.calls.empty(), n + ": a none choice makes no call"); }},
            {"tools serial", {kWeather, kSearchStrict}, std::nullopt,
             ToolChoice{.mode = ToolChoiceMode::Required, .parallel = false}, {},
             [&](const Walk& w, const std::string& n) { check(w.calls.size() == 1, n + ": parallel false calls once"); }},
            {"json beside tools", {kWeather}, OutputConstraint::json_schema(schema), basic(ToolChoiceMode::Auto), {},
             [&](const Walk& w, const std::string& n) {
                 if (w.calls.empty()) {
                     const JsonShape shape = json_shape(w.content);
                     check(shape.ok && shape.type == "object", n + ": the answer is the JSON: " + w.content);
                 } else {
                     check(w.content.empty(), n + ": calls come without a text answer");
                     callable(w, n, {"get_weather"});
                 }
             }},
        };
        std::mt19937 rng(20261124);
        for (const Case& test : cases) {
            const ink::PreparedPrompt prompt = frontend.prepare(question(test.tools));
            std::size_t ended = 0, walks = 0, steps = 0, applied = 0;
            for (int seed = 0; seed < 24; ++seed) {
                ink::OutputSession session =
                    frontend.make_output_session(prompt, {}, {}, test.thinking, test.constraint, test.choice);
                check(session.constrained(), test.name + ": the session is constrained");
                const bool may_end_empty = !test.constraint && (test.choice.mode == ToolChoiceMode::None ||
                                                                (test.choice.mode == ToolChoiceMode::Auto &&
                                                                 !test.choice.allowed_names));
                const Walk w = walk(c, session, rng, 3000, test.name, may_end_empty);
                ++walks;
                steps += w.steps;
                applied += w.applied ? 1U : 0U;
                if (w.finish == FinishReason::StopToken) {
                    ++ended;
                    test.ended(w, test.name);
                    const auto observed = session.constraint_observation();
                    check(observed && observed->complete && observed->terminated, test.name + ": the matcher terminated");
                }
            }
            std::cout << test.name << ": " << ended << " of " << walks << " walks ended at the stop token, "
                      << steps / walks << " tokens a walk" << (applied ? ", " + std::to_string(applied) + " budget controls" : "")
                      << '\n';
            check(ended * 2 >= walks, test.name + ": most walks end");
        }

        // The free route: an auto choice of non-strict tools under Automatic, and no tools.
        {
            const ink::PreparedPrompt tools = frontend.prepare(question({kWeather}));
            check(!frontend.make_output_session(tools, {}, {}, {}, {}, ToolChoice{.constraints = ToolConstraintMode::Automatic})
                       .constrained(),
                  "an automatic auto choice of non-strict tools is free");
            check(frontend.make_output_session(tools, {}, {}, {}, {}, basic(ToolChoiceMode::Auto)).constrained(),
                  "a basic auto choice is constrained");
            const ink::PreparedPrompt plain = frontend.prepare(question());
            check(!frontend.make_output_session(plain, {}).constrained(), "no tools and no constraint is free");
        }

        // 3. Refusals.
        {
            const ink::PreparedPrompt plain = frontend.prepare(question());
            const ink::PreparedPrompt tools = frontend.prepare(question({kWeather}));
            const auto refused = [&](const std::function<void()>& body, infernix::RequestErrorKind kind) {
                try {
                    body();
                } catch (const infernix::RequestError& error) { return error.kind() == kind; }
                return false;
            };
            check(refused([&] {
                      (void)frontend.make_output_session(plain, {.strings = {{.text = "END"}}}, {}, {},
                                                         OutputConstraint::json_object());
                  },
                          infernix::RequestErrorKind::InvalidJsonSchema),
                  "a stop string beside a constraint is refused");
            check(refused([&] { (void)frontend.make_output_session(plain, {}, {.raw = true}, {}, OutputConstraint::regex("a")); },
                          infernix::RequestErrorKind::InvalidRegex),
                  "raw output beside a constraint is refused");
            check(refused([&] {
                      (void)frontend.make_output_session(tools, {}, {}, {}, OutputConstraint::regex("a"), basic(ToolChoiceMode::Auto));
                  },
                          infernix::RequestErrorKind::InvalidRegex),
                  "a regex answer beside callable tools is refused");
            check(refused([&] {
                      (void)frontend.make_output_session(
                          tools, {}, {}, {}, {},
                          ToolChoice{.mode = ToolChoiceMode::Required, .allowed_names = std::vector<std::string>{"nope"}});
                  },
                          infernix::RequestErrorKind::InvalidToolConstraint),
                  "a named choice of an undeclared tool is refused");
            check(refused([&] { (void)frontend.make_output_session(plain, {}, {}, {}, {}, basic(ToolChoiceMode::Required)); },
                          infernix::RequestErrorKind::InvalidToolConstraint),
                  "a required choice without tools is refused");
            check(refused([&] {
                      (void)frontend.make_output_session(
                          plain, {}, {}, {},
                          OutputConstraint::grammar("root ::= Token(" + std::to_string(c.markers[6]) + ")"));
                  },
                          infernix::RequestErrorKind::InvalidGrammar),
                  "a GBNF that names a marker is refused");
        }

        // 4. Mask cost.
        for (auto& [kind, values] : c.costs.us) {
            std::sort(values.begin(), values.end());
            double sum = 0;
            for (const double v : values) { sum += v; }
            std::cout << "mask " << kind << ": " << values.size() << " masks, mean " << sum / values.size() << " us, median "
                      << values[values.size() / 2] << " us, p95 " << values[values.size() * 95 / 100] << " us, max "
                      << values.back() << " us\n";
        }
        std::sort(c.costs.first_wait_us.begin(), c.costs.first_wait_us.end());
        std::cout << "first mask (compile wait included): median " << c.costs.first_wait_us[c.costs.first_wait_us.size() / 2]
                  << " us, max " << c.costs.first_wait_us.back() << " us\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "inkling2 constraints: all checks passed\n";
    return 0;
}
