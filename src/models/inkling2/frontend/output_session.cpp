#include "models/inkling2/frontend/output_session.h"

#include "models/output_text.h"
#include "runtime/contract/execution.h"
#include "text/grammar.h"
#include "text/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace infernix::models::inkling2 {
namespace {

// Where the turn is: between blocks (headers, stray text and the end-of-turn token publish nothing), or inside a
// thinking, content or tool-call block until its <|end_message|>.
enum class Phase : std::uint8_t { Between, Reasoning, Content, Call };

std::size_t channel_index(OutputChannel channel) noexcept { return channel == OutputChannel::Reasoning ? 0 : 1; }

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

// The earliest match wins: by token, then by byte, then by declaration (as Qwen's sessions order them).
bool precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut, std::uint32_t order, const StopMatch& current) {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) { return committed_tokens < current.committed_tokens; }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return order < current.declaration_order;
}

// Everything one round's preview copies: the decoder's position, the thinking budget's accounting and the calls the
// turn has completed.
struct State {
    Phase phase = Phase::Between;
    std::string utf8_pending;                // an incomplete code point of the open block
    std::array<std::string, 2> stop_pending; // per channel: bytes that may start a stop string
    std::uint64_t decoded_bytes = 0;         // bytes fed to the channels (stop-match order)
    std::string call_body;                   // the open call's bytes
    bool terminal                  = false;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool control_pending                = false;
    bool applied                        = false; // the control closed the thinking
    // A thinking block closed on its own with the budget used up: the budget is spent, as an applied control spends
    // it (answer rounds are not held to one token, and reasoning_ban refuses another thinking block for the turn).
    bool spent = false;
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
    // A constrained turn's branch (ConstraintObservation::branch): it opened a content block, or a call block.
    bool opened_content = false;
    bool opened_call    = false;
    // A <|content_text|> block that may be calls in an XML notation (Settings::notation_tools): Open while the block's
    // text begins (or may still begin) with a notation opener, its output held in notation_output and its text kept;
    // Closed once the closed block parsed as calls (notation_calls), until what follows decides. A block that is not
    // one publishes what it held, as it would have unheld (stop strings matched it as it came).
    enum class Notation : std::uint8_t { None, Open, Closed };
    Notation notation = Notation::None;
    std::string notation_text;
    PublishedOutput notation_output;
    std::vector<GeneratedToolCall> notation_calls;
};

// The block openers of the XML notation (JC-01).
constexpr std::string_view kNotationOpeners[] = {"<tool_call>", "<function_calls>", "<invoke "};

std::size_t skip_space(std::string_view s, std::size_t at) {
    while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\n' || s[at] == '\r')) { ++at; }
    return at;
}

// Whether a text block's text so far may still be, or begins with, a notation opener after its leading whitespace.
bool notation_candidate(std::string_view text) {
    const std::string_view t = text.substr(skip_space(text, 0));
    for (const std::string_view opener : kNotationOpeners) {
        if (t.starts_with(opener) || opener.starts_with(t)) { return true; }
    }
    return false;
}

// One `<invoke>` of the notation: the tool's name and its parameters (name, value text, the string="..." attribute or
// empty).
struct NotationInvoke {
    std::string name;
    struct Parameter {
        std::string name, value, string_flag;
    };
    std::vector<Parameter> parameters;
};

// A quoted attribute `KEY="VALUE"` (or single-quoted) at `at` after whitespace; VALUE has no quote, `<` or `>`.
bool notation_attribute(std::string_view s, std::size_t& at, std::string_view key, std::string& value) {
    std::size_t i = skip_space(s, at);
    if (i == at || !s.substr(i).starts_with(key) || i + key.size() >= s.size() || s[i + key.size()] != '=') { return false; }
    i += key.size() + 1;
    if (i >= s.size() || (s[i] != '"' && s[i] != '\'')) { return false; }
    const char quote = s[i++];
    const std::size_t end = s.find(quote, i);
    if (end == std::string_view::npos) { return false; }
    value = std::string(s.substr(i, end - i));
    if (value.find_first_of("\"'<>") != std::string::npos) { return false; }
    at = end + 1;
    return true;
}

// The whole of `s` as notation calls: optional `<tool_call>` and `<function_calls>` wrappers (each closed if opened), one
// or more closed invokes, whitespace between; anything else (text, an unclosed tag, a tag left open) is not. A value
// ends at the `</parameter>` that whitespace and then `<parameter ` or `</invoke>` follow (one quoted inside it is the
// value's), as Qwen's parser and Basalt's read a raw value.
bool parse_notation(std::string_view s, std::vector<NotationInvoke>& calls) {
    std::size_t at = skip_space(s, 0);
    const auto take = [&](std::string_view tag) {
        if (!s.substr(at).starts_with(tag)) { return false; }
        at = skip_space(s, at + tag.size());
        return true;
    };
    const bool tool_call      = take("<tool_call>");
    const bool function_calls = take("<function_calls>");
    while (s.substr(at).starts_with("<invoke")) {
        NotationInvoke call;
        at += std::string_view("<invoke").size();
        if (!notation_attribute(s, at, "name", call.name) || call.name.empty()) { return false; }
        at = skip_space(s, at);
        if (at >= s.size() || s[at] != '>') { return false; }
        at = skip_space(s, at + 1);
        for (;;) {
            if (take("</invoke>")) { break; }
            if (!s.substr(at).starts_with("<parameter")) { return false; }
            NotationInvoke::Parameter p;
            at += std::string_view("<parameter").size();
            if (!notation_attribute(s, at, "name", p.name) || p.name.empty()) { return false; }
            std::size_t after = at;
            if (notation_attribute(s, after, "string", p.string_flag)) {
                if (p.string_flag != "true" && p.string_flag != "false") { return false; }
                at = after;
            }
            at = skip_space(s, at);
            if (at >= s.size() || s[at] != '>') { return false; }
            const std::size_t start = ++at;
            for (std::size_t from = start;;) {
                const std::size_t close = s.find("</parameter>", from);
                if (close == std::string_view::npos) { return false; }
                const std::size_t next = skip_space(s, close + std::string_view("</parameter>").size());
                const std::string_view rest = s.substr(next);
                if (rest.starts_with("<parameter ") || rest.starts_with("</invoke>")) {
                    p.value = std::string(s.substr(start, close - start));
                    at      = next;
                    break;
                }
                from = close + 1;
            }
            for (const NotationInvoke::Parameter& seen : call.parameters) {
                if (seen.name == p.name) { return false; } // a repeated parameter: not one call's arguments
            }
            call.parameters.push_back(std::move(p));
        }
        calls.push_back(std::move(call));
    }
    if (calls.empty()) { return false; }
    if (function_calls && !take("</function_calls>")) { return false; }
    if (tool_call && !take("</tool_call>")) { return false; }
    return at == s.size();
}

void feed_channel(State& state, OutputChannel channel, std::string_view text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best) {
    if (text.empty()) { return; }
    std::string& pending          = state.stop_pending[channel_index(channel)];
    std::string combined          = pending;
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;
    if (best != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel || stop.text.empty()) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const auto order = static_cast<std::uint32_t>(declaration);
            if (!precedes(committed_tokens, combined_start + found, order, *best)) { continue; }
            PublishedOutput candidate = emitted;
            candidate.append(channel, std::string_view(combined).substr(0, found));
            if (stop.include_in_output) { candidate.append(channel, stop.text); }
            *best = StopMatch{.found             = true,
                              .committed_tokens  = committed_tokens,
                              .byte_cut          = combined_start + found,
                              .declaration_order = order,
                              .output            = std::move(candidate)};
        }
    }
    const std::size_t hold = stop_hold_size(combined, channel, policy);
    emitted.append(channel, std::string_view(combined).substr(0, combined.size() - hold));
    pending = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

// Publishes what a channel held for a stop string's possible start: a block's close ends the text a stop string can
// span.
void flush_channel(State& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    emitted.append(channel, pending);
    pending.clear();
}

// A call's JSON read for its syntax alone: each value's text is kept as written, so a number keeps its digits whatever
// its range (a grammar admits any JSON number; a JSON library would round a 20-digit integer or refuse 1e400).
class CallJson {
public:
    explicit CallJson(std::string_view text) : s_(text) {}

    // The top-level object's "name" string token and "args" value text (the last of a repeated member); false when the
    // text is not one JSON object.
    bool members(std::string_view& name, std::string_view& args) {
        ws();
        if (!take('{')) { return false; }
        ws();
        if (!take('}')) {
            for (;;) {
                ws();
                const std::size_t key_at = i_;
                if (!string()) { return false; }
                const std::string_view key_token = s_.substr(key_at, i_ - key_at);
                ws();
                if (!take(':')) { return false; }
                ws();
                const std::size_t value_at = i_;
                if (!value(0)) { return false; }
                const std::string_view text = s_.substr(value_at, i_ - value_at);
                if (key_token == "\"name\"") { name = text; }
                if (key_token == "\"args\"") { args = text; }
                ws();
                if (take(',')) { continue; }
                if (take('}')) { break; }
                return false;
            }
        }
        ws();
        return i_ == s_.size();
    }

    // Whether the text is one JSON value (whitespace around it allowed).
    bool whole_value() {
        ws();
        if (!value(0)) { return false; }
        ws();
        return i_ == s_.size();
    }

private:
    static constexpr int kMaximumDepth = 256;

    bool take(char c) {
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) { ++i_; }
    }
    bool digits() {
        const std::size_t start = i_;
        while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; }
        return i_ > start;
    }
    bool string() {
        if (!take('"')) { return false; }
        for (; i_ < s_.size(); ++i_) {
            const auto c = static_cast<unsigned char>(s_[i_]);
            if (c == '"') {
                ++i_;
                return true;
            }
            if (c < 0x20) { return false; }
            if (c != '\\') { continue; }
            if (++i_ >= s_.size()) { return false; }
            if (s_[i_] == 'u') {
                for (int k = 0; k < 4; ++k) {
                    if (++i_ >= s_.size() || std::isxdigit(static_cast<unsigned char>(s_[i_])) == 0) { return false; }
                }
            } else if (std::string_view("\"\\/bfnrt").find(s_[i_]) == std::string_view::npos) {
                return false;
            }
        }
        return false;
    }
    bool value(int depth) {
        if (i_ >= s_.size() || depth > kMaximumDepth) { return false; }
        const char c = s_[i_];
        if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            ++i_;
            ws();
            if (take(close)) { return true; }
            for (;;) {
                ws();
                if (c == '{') {
                    if (!string()) { return false; }
                    ws();
                    if (!take(':')) { return false; }
                    ws();
                }
                if (!value(depth + 1)) { return false; }
                ws();
                if (take(',')) { continue; }
                return take(close);
            }
        }
        if (c == '"') { return string(); }
        for (const std::string_view word : {"true", "false", "null"}) {
            if (s_.substr(i_, word.size()) == word) {
                i_ += word.size();
                return true;
            }
        }
        take('-');
        if (!take('0') && !digits()) { return false; }
        if (take('.') && !digits()) { return false; }
        if (take('e') || take('E')) {
            if (!take('+')) { take('-'); }
            if (!digits()) { return false; }
        }
        return true;
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

// The JSON types a property of `schema` (the tool's parameter schema, JSON text) declares: its "type" (a name or an
// array of names), else the types of its enum / const values; empty when it declares none.
std::vector<std::string> declared_types(const nlohmann::ordered_json& schema, const std::string& property) {
    std::vector<std::string> types;
    if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object() ||
        !schema["properties"].contains(property) || !schema["properties"][property].is_object()) {
        return types;
    }
    const auto& p = schema["properties"][property];
    const auto add_value_type = [&](const nlohmann::ordered_json& v) {
        types.push_back(v.is_string() ? "string" : v.is_boolean() ? "boolean" : v.is_number() ? "number"
                        : v.is_null() ? "null" : v.is_array() ? "array" : "object");
    };
    if (p.contains("type") && p["type"].is_string()) {
        types.push_back(p["type"].get<std::string>());
    } else if (p.contains("type") && p["type"].is_array()) {
        for (const auto& t : p["type"]) {
            if (t.is_string()) { types.push_back(t.get<std::string>()); }
        }
    } else if (p.contains("enum") && p["enum"].is_array()) {
        for (const auto& v : p["enum"]) { add_value_type(v); }
    } else if (p.contains("const")) {
        add_value_type(p["const"]);
    }
    return types;
}

// The JSON type of a value text CallJson accepted.
std::string value_type(std::string_view text) {
    const std::size_t at = skip_space(text, 0);
    const char c         = at < text.size() ? text[at] : '\0';
    if (c == '{') { return "object"; }
    if (c == '[') { return "array"; }
    if (c == '"') { return "string"; }
    if (c == 't' || c == 'f') { return "boolean"; }
    if (c == 'n') { return "null"; }
    return "number";
}

// A notation call's arguments as JSON text: a value marked string="true" is its text; one marked string="false" is
// the JSON it holds (its text when it holds none); otherwise the value is the JSON it holds when that is a type its
// property declares (any type when it declares none, a string never: a string value is its text) and its text
// otherwise. JSON values keep their text as written (CallJson), so a number keeps its digits. No entity is decoded:
// the value is the model's text. False when a value is not valid UTF-8 text.
bool notation_arguments(const NotationInvoke& call, const DeclaredTool& tool, std::string& out) {
    const nlohmann::ordered_json schema = nlohmann::ordered_json::parse(tool.parameters, nullptr, false);
    out = "{";
    try {
        for (const NotationInvoke::Parameter& p : call.parameters) {
            if (out.size() > 1) { out.push_back(','); }
            out += nlohmann::json(p.name).dump();
            out.push_back(':');
            const bool json = p.string_flag != "true" && CallJson(p.value).whole_value();
            const std::vector<std::string> types =
                p.string_flag.empty() ? declared_types(schema, p.name) : std::vector<std::string>{};
            const std::string type = json ? value_type(p.value) : std::string();
            const bool admitted    = json && (type != "string" || p.string_flag == "false") &&
                                  (types.empty() || std::find(types.begin(), types.end(), type) != types.end() ||
                                   (type == "number" && std::find(types.begin(), types.end(), "integer") != types.end()));
            if (admitted) {
                const std::size_t first = skip_space(p.value, 0);
                std::size_t last        = p.value.size();
                while (last > first && std::string_view(" \t\n\r").find(p.value[last - 1]) != std::string_view::npos) {
                    --last;
                }
                out.append(p.value, first, last - first);
            } else {
                out += nlohmann::json(p.value).dump();
            }
        }
    } catch (const nlohmann::json::exception&) {
        return false; // not valid UTF-8
    }
    out.push_back('}');
    return true;
}

} // namespace

void PublishedOutput::append(OutputChannel channel, std::string_view text) {
    if (text.empty()) { return; }
    if (!values_.empty() && values_.back().channel == channel) {
        values_.back().text.append(text);
    } else {
        values_.push_back(OutputDelta{.channel = channel, .text = std::string(text)});
    }
}

class OutputSession::Impl {
public:
    explicit Impl(Settings settings) : s(std::move(settings)) {
        if (s.tokenizer == nullptr) { throw std::invalid_argument("inkling2 output session: no tokenizer"); }
        if (s.thinking.budget && *s.thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        if (s.thinking.loop != ReasoningLoopAction::Off) {
            throw std::invalid_argument("the reasoning-loop guard is not available for Inkling 2 yet");
        }
        if (s.thinking_control == nullptr || s.thinking_control->empty() ||
            s.thinking_control->back() != s.markers.end_message) {
            throw std::invalid_argument("inkling2 output session: the thinking control must end with <|end_message|>");
        }
        raw            = s.output.raw;
        preserve       = s.output.raw || s.output.preserve_special_tokens;
        parse_calls    = !raw && s.tools != nullptr;
        notation       = parse_calls && !s.notation_tools.empty();
        // An opened answer: the turn starts inside its text block.
        if (s.answer_opened && !raw) {
            committed.phase          = Phase::Content;
            committed.opened_content = true;
        }
    }

    [[nodiscard]] bool stop_token(TokenId token) const {
        return std::find(s.policy.token_ids.begin(), s.policy.token_ids.end(), token) != s.policy.token_ids.end();
    }

    [[nodiscard]] bool content_opener(TokenId token) const {
        return std::find(s.markers.content.begin(), s.markers.content.end(), token) != s.markers.content.end() ||
               (token == s.markers.tool_call && !parse_calls);
    }

    // Decodes the open block's bytes into its channel.
    void feed_text(State& state, OutputChannel channel, std::string_view bytes, PublishedOutput& emitted,
                   std::uint32_t committed, StopMatch* best) const {
        state.utf8_pending.append(bytes);
        const std::string text = consume_generated_utf8(state.utf8_pending);
        feed_channel(state, channel, text, s.policy, emitted, committed, best);
    }

    void close_text(State& state, OutputChannel channel, PublishedOutput& emitted, std::uint32_t committed,
                    StopMatch* best) const {
        if (!state.utf8_pending.empty()) { // the close cuts a code point
            state.utf8_pending.clear();
            feed_channel(state, channel, kUtf8Replacement, s.policy, emitted, committed, best);
        }
        flush_channel(state, channel, emitted);
    }

    // A call that is not one becomes the turn's text, unmatched (a call's bytes are the tool's data).
    void fall_back(State& state, std::string body, ToolCallParseFallbackReason reason, PublishedOutput& emitted) const {
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = reason;
        }
        static const StopPolicy kNoStops{};
        std::string text = consume_generated_utf8(body);
        if (!body.empty()) { text.append(kUtf8Replacement); }
        flush_channel(state, OutputChannel::Content, emitted);
        feed_channel(state, OutputChannel::Content, text, kNoStops, emitted, 0, nullptr);
    }

    // The response_template's tool_calls field: {"name": NAME, "args": {...}} becomes the call NAME(args), its arguments
    // the args object's text as the model wrote it (CallJson: every number as written).
    void close_call(State& state, PublishedOutput& emitted) const {
        std::string body = std::move(state.call_body);
        state.call_body.clear();
        std::string_view name_token, args;
        const bool utf8 = [&] {
            try {
                (void)text::unicode_internal::utf8_codepoints(body, "tool call");
                return true;
            } catch (const std::exception&) { return false; }
        }();
        if (!utf8 || !CallJson(body).members(name_token, args) || args.empty() || args.front() != '{' ||
            name_token.empty()) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::MalformedStructure, emitted);
            return;
        }
        std::string tool;
        if (name_token.front() == '"') {
            try {
                tool = nlohmann::json::parse(name_token).get<std::string>();
            } catch (const nlohmann::json::exception&) { tool.clear(); } // a lone surrogate's escape
        }
        if (tool.empty()) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::InvalidToolName, emitted);
            return;
        }
        if (!s.tools->open && !s.tools->declared.contains(tool)) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::UndeclaredTool, emitted);
            return;
        }
        const auto declared = std::find_if(s.tools->tools.begin(), s.tools->tools.end(),
                                           [&](const DeclaredTool& entry) { return entry.name == tool; });
        if (declared != s.tools->tools.end() && omits_required(args, declared->required)) {
            ++state.diagnostics.required_arguments_missing;
        }
        state.calls.push_back(GeneratedToolCall{.name = std::move(tool), .arguments_json = std::string(args)});
        ++state.diagnostics.structured_call_count;
    }

    // A notation candidate's text (State::Notation::Open): its output goes to the hold, its text is kept, and a stop
    // string still matches it as it comes (the match's output is what was published before the hold, then the held
    // text to the stop). `close`: the block's close (an incomplete code point becomes U+FFFD, the stop hold flushes).
    void feed_held(State& state, std::string_view bytes, bool close, PublishedOutput& emitted, std::uint32_t committed,
                   StopMatch* best) const {
        StopMatch local;
        StopMatch* const match = best != nullptr ? &local : nullptr;
        if (close) {
            if (!state.utf8_pending.empty()) { state.notation_text.append(kUtf8Replacement); }
            close_text(state, OutputChannel::Content, state.notation_output, committed, match);
        } else {
            state.utf8_pending.append(bytes);
            const std::string text = consume_generated_utf8(state.utf8_pending);
            state.notation_text += text;
            feed_channel(state, OutputChannel::Content, text, s.policy, state.notation_output, committed, match);
        }
        if (match != nullptr && local.found &&
            precedes(local.committed_tokens, local.byte_cut, local.declaration_order, *best)) {
            PublishedOutput output = emitted;
            for (const OutputDelta& delta : local.output) { output.append(delta.channel, delta.text); }
            local.output = std::move(output);
            *best        = std::move(local);
        }
    }

    // A held block that is not calls publishes what it held, where it would have.
    void release_notation(State& state, PublishedOutput& emitted) const {
        for (const OutputDelta& delta : state.notation_output) { emitted.append(delta.channel, delta.text); }
        state.notation = State::Notation::None;
        state.notation_text.clear();
        state.notation_output.clear();
        state.notation_calls.clear();
    }

    // A closed held block followed by a call block or the turn's end: its calls are the turn's.
    void accept_notation(State& state) const {
        for (GeneratedToolCall& call : state.notation_calls) {
            const auto declared = std::find_if(s.notation_tools.begin(), s.notation_tools.end(),
                                               [&](const DeclaredTool& tool) { return tool.name == call.name; });
            if (declared != s.notation_tools.end() && omits_required(call.arguments_json, declared->required)) {
                ++state.diagnostics.required_arguments_missing;
            }
            state.calls.push_back(std::move(call));
            ++state.diagnostics.structured_call_count;
            ++state.diagnostics.notation_calls;
        }
        state.diagnostics.marker_seen = true;
        state.notation                = State::Notation::None;
        state.notation_text.clear();
        state.notation_output.clear();
        state.notation_calls.clear();
    }

    // A held block's close: whole notation calls of the turn's notation tools wait for what follows (Closed); anything
    // else is text.
    void close_notation(State& state, PublishedOutput& emitted) const {
        std::vector<NotationInvoke> invokes;
        std::vector<GeneratedToolCall> calls;
        bool ok = parse_notation(state.notation_text, invokes);
        for (std::size_t i = 0; ok && i < invokes.size(); ++i) {
            const auto declared = std::find_if(s.notation_tools.begin(), s.notation_tools.end(),
                                               [&](const DeclaredTool& tool) { return tool.name == invokes[i].name; });
            std::string arguments;
            ok = declared != s.notation_tools.end() && notation_arguments(invokes[i], *declared, arguments);
            if (ok) { calls.push_back(GeneratedToolCall{.name = invokes[i].name, .arguments_json = std::move(arguments)}); }
        }
        if (ok && !s.parallel_calls && state.calls.size() + calls.size() > 1) { ok = false; }
        if (!ok) {
            release_notation(state, emitted);
            return;
        }
        state.notation       = State::Notation::Closed;
        state.notation_calls = std::move(calls);
        state.notation_text.clear();
    }

    // One generated (or injected) token.
    void feed_token(State& state, TokenId token, PublishedOutput& emitted, std::uint32_t committed,
                    StopMatch* best) const {
        const text::DecodedTokenView decoded = s.tokenizer->decoded_token(token);
        if (raw) {
            feed_text(state, OutputChannel::Content, decoded.bytes, emitted, committed, best);
            return;
        }
        if (state.phase == Phase::Between && state.notation == State::Notation::Closed) {
            // What follows a closed notation block decides it: a call block (when the turn may make several calls)
            // keeps it calls; a thinking or text block makes it text. Headers between blocks decide nothing.
            if (token == s.markers.tool_call && parse_calls) {
                if (s.parallel_calls) {
                    accept_notation(state);
                } else {
                    release_notation(state, emitted);
                }
            } else if (token == s.markers.thinking || content_opener(token)) {
                release_notation(state, emitted);
            }
        }
        switch (state.phase) {
        case Phase::Between:
            if (token == s.markers.thinking) {
                state.phase = Phase::Reasoning;
            } else if (content_opener(token)) {
                state.phase          = Phase::Content;
                state.opened_content = true;
                if (notation && token == s.markers.content[0]) { // a <|content_text|> block
                    state.notation = State::Notation::Open;
                    state.notation_text.clear();
                    state.notation_output.clear();
                }
            } else if (token == s.markers.tool_call) {
                state.phase                   = Phase::Call;
                state.opened_call             = true;
                state.diagnostics.marker_seen = true;
            }
            return; // headers, stray text and other tokens between blocks publish nothing
        case Phase::Reasoning:
        case Phase::Content: {
            const OutputChannel channel =
                state.phase == Phase::Reasoning ? OutputChannel::Reasoning : OutputChannel::Content;
            const bool held = state.phase == Phase::Content && state.notation == State::Notation::Open;
            if (token == s.markers.end_message) {
                if (held) {
                    feed_held(state, {}, true, emitted, committed, best);
                    close_notation(state, emitted);
                } else {
                    close_text(state, channel, emitted, committed, best);
                }
                state.phase = Phase::Between;
                return;
            }
            if (decoded.special && !preserve) { return; }
            if (held) {
                feed_held(state, decoded.bytes, false, emitted, committed, best);
                // Text that can no longer begin with an opener publishes what was held and goes on unheld.
                if (!notation_candidate(state.notation_text)) { release_notation(state, emitted); }
                return;
            }
            feed_text(state, channel, decoded.bytes, emitted, committed, best);
            return;
        }
        case Phase::Call:
            if (token == s.markers.end_message) {
                close_call(state, emitted);
                state.phase = Phase::Between;
                return;
            }
            if (decoded.special && !preserve) { return; }
            state.call_body.append(decoded.bytes);
            return;
        }
    }

    // The turn ends with a block open (a limit, a cancellation, a stop token): its text publishes; an open call is
    // text. `turn_ended`: a stop token ended the turn, which keeps a closed notation block calls; a limit or a
    // cancellation leaves it text (the turn did not end after it), as a notation block it cut.
    void terminalize(State& state, PublishedOutput& emitted, bool turn_ended = false) const {
        if (state.notation == State::Notation::Open && state.phase == Phase::Content) {
            feed_held(state, {}, true, emitted, 0, nullptr);
            release_notation(state, emitted);
        } else if (state.notation == State::Notation::Closed) {
            if (turn_ended) {
                accept_notation(state);
            } else {
                release_notation(state, emitted);
            }
        }
        if (state.phase == Phase::Reasoning || raw) {
            close_text(state, raw ? OutputChannel::Content : OutputChannel::Reasoning, emitted, 0, nullptr);
        } else if (state.phase == Phase::Content) {
            close_text(state, OutputChannel::Content, emitted, 0, nullptr);
        } else if (state.phase == Phase::Call) {
            std::string body = std::move(state.call_body);
            state.call_body.clear();
            fall_back(state, std::move(body), ToolCallParseFallbackReason::MalformedStructure, emitted);
        }
        flush_channel(state, OutputChannel::Reasoning, emitted);
        flush_channel(state, OutputChannel::Content, emitted);
        state.utf8_pending.clear();
        state.phase    = Phase::Between;
        state.terminal = true;
    }

    void begin_preview() {
        preview       = committed;
        preview_ready = false;
        preview_output.clear();
    }

    // The matcher takes a round's accepted tokens (tentatively, until the commit). A refusal fails this request only
    // (runtime::RequestOutputError): its sampler drew only legal tokens, so one is a defect, never the model's choice.
    void accept_grammar(std::span<const TokenId> tokens) const {
        if (!s.grammar) { return; }
        try {
            s.grammar->accept(tokens);
        } catch (const std::exception& error) {
            s.grammar->discard();
            throw runtime::RequestOutputError(std::string("constrained output: ") + error.what());
        }
    }

    Settings s;
    bool raw         = false;
    bool preserve    = false;
    bool parse_calls = false;
    bool notation    = false; // Settings::notation_tools applies
    State committed;
    State preview;
    PublishedOutput preview_output;
    bool preview_ready                 = false;
    FinishReason preview_finish_reason = FinishReason::None;
    std::vector<GeneratedToolCall> tool_calls; // the terminal turn's calls
};

OutputSession::OutputSession() noexcept = default;
OutputSession::OutputSession(Settings settings) : impl_(std::make_unique<Impl>(std::move(settings))) {}
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining, FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (m.committed.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) { throw std::invalid_argument("cannot preview an empty generated-token round"); }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit && limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }
    m.begin_preview();
    State& state = m.preview;
    const std::optional<std::uint32_t> budget = m.s.thinking.budget;

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation = runtime::ContinuationAction::Decode) {
        m.accept_grammar(tokens.first(count));
        m.preview_finish_reason = reason;
        m.preview_ready         = true;
        return runtime::OutputDecision{.accepted_tokens = count, .finish_reason = reason, .continuation = continuation};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto count    = static_cast<std::uint32_t>(index + 1);
        const TokenId token = tokens[index];
        const bool thinking = !m.raw && state.phase == Phase::Reasoning;
        if (thinking) {
            ++state.reasoning_tokens;
            ++state.model_thinking_tokens;
            if (budget && !state.applied && !state.spent && state.model_thinking_tokens > *budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
        }
        if (m.stop_token(token)) {
            if (m.s.policy.publish_stop_token) { m.feed_token(state, token, m.preview_output, count, nullptr); }
            m.terminalize(state, m.preview_output, true);
            return complete(count, FinishReason::StopToken);
        }
        StopMatch match;
        m.feed_token(state, token, m.preview_output, count, &match);
        if (thinking && budget && state.phase != Phase::Reasoning && state.model_thinking_tokens >= *budget) {
            state.spent = true;
        }
        if (match.found) {
            state.matched_stop_order = match.declaration_order;
            state.utf8_pending.clear();
            state.stop_pending = {};
            state.call_body.clear();
            state.phase    = Phase::Between;
            state.terminal = true;
            m.preview_output = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }
    }
    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        m.terminalize(state, m.preview_output);
        return complete(count, limit_reason);
    }
    if (budget && !state.applied && !state.spent && state.phase == Phase::Reasoning &&
        state.model_thinking_tokens == *budget) {
        // The control and one model token after it must fit the output budget; otherwise the turn ends at its limit.
        const std::uint64_t remaining = total_budget_remaining - count;
        if (remaining < m.s.thinking_control->size() + 1U) {
            m.terminalize(state, m.preview_output);
            return complete(count, FinishReason::OutputLimit);
        }
        state.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr) { return total_budget_remaining; }
    const State& state = impl_->committed;
    if (state.control_pending) { return 0; }
    const std::optional<std::uint32_t> budget = impl_->s.thinking.budget;
    if (!budget || state.applied || state.spent || impl_->raw) { return total_budget_remaining; }
    // A round may open a thinking block (its opener is not a thinking token) and then fill what is left of the
    // budget: the budget is reached at a round's last token at the latest.
    const std::uint32_t left = *budget - std::min(*budget, state.model_thinking_tokens);
    const std::uint32_t cap  = state.phase == Phase::Reasoning ? left : left + 1U;
    return std::min(total_budget_remaining, cap);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->committed.control_pending) { return {}; }
    return *impl_->s.thinking_control;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || !std::equal(tokens.begin(), tokens.end(), expected.begin(), expected.end())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }
    m.begin_preview();
    State& state = m.preview;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (state.phase == Phase::Reasoning) { ++state.reasoning_tokens; }
        m.feed_token(state, tokens[index], m.preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (state.phase != Phase::Between) {
        throw std::logic_error("the thinking control did not close the thinking block");
    }
    m.accept_grammar(tokens);
    state.control_pending   = false;
    state.applied           = true;
    state.injected_tokens   = static_cast<std::uint32_t>(tokens.size());
    m.preview_finish_reason = FinishReason::None;
    m.preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = static_cast<std::uint32_t>(tokens.size())};
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    const std::optional<std::uint32_t> budget = impl_->s.thinking.budget;
    if (!budget || impl_->raw || effective_output_tokens <= *budget) { return; }
    // The budget is reached after its block's two-token header (<|message_model|><|content_thinking|>) and *budget
    // thinking tokens; the control and one model token follow.
    const std::uint64_t required = static_cast<std::uint64_t>(*budget) + 2U + impl_->s.thinking_control->size() + 1U;
    if (effective_output_tokens < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control suffix and one "
            "post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString || reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    m.begin_preview();
    m.preview.control_pending = false;
    m.terminalize(m.preview, m.preview_output);
    m.preview_finish_reason = reason;
    m.preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    // A commit without a preview is the Engine's own protocol error: thrown, so the commit's caller fails this
    // request (EngineCore::commit_output) or the worker answers every request, rather than the process aborting.
    if (impl_ == nullptr || !impl_->preview_ready) {
        throw std::logic_error("Inkling 2 output session: commit_preview without a preview to commit");
    }
    Impl& m = *impl_;
    std::swap(m.committed, m.preview);
    m.preview_ready       = false;
    PublishedOutput output = std::move(m.preview_output);
    m.preview_output.clear();
    if (m.s.grammar) { m.s.grammar->confirm(); }
    if (m.committed.terminal) { m.tool_calls = std::move(m.committed.calls); }
    return output;
}

void OutputSession::discard_preview() {
    if (impl_ == nullptr) { return; }
    if (impl_->s.grammar) { impl_->s.grammar->discard(); }
    impl_->preview_ready = false;
    impl_->preview_output.clear();
}

runtime::ReasoningBan OutputSession::reasoning_ban() const noexcept {
    if (impl_ == nullptr || impl_->raw || impl_->committed.terminal) { return {}; }
    const State& state = impl_->committed;
    // A thinking budget holds for the whole turn: once its control closed the thinking (applied) or a block closed with
    // the budget used up (spent), the model may not open another thinking block, which would think uncapped to the
    // output limit. Every later column refuses the thinking opener. The ban never lifts: its close is the opener itself,
    // which no column can draw (a drafted opener in a verify column's overlay is a draft that column 0 already refused).
    if (impl_->s.thinking.budget && (state.applied || state.spent) && state.phase != Phase::Reasoning) {
        return {.close = impl_->s.markers.thinking, .ends = {impl_->s.markers.thinking, -1}, .withheld = -1};
    }
    if (impl_->s.thinking.end_turn_in_thinking || state.phase != Phase::Reasoning) { return {}; }
    return {.close    = impl_->s.markers.end_message,
            .ends     = {impl_->s.markers.end_of_turn, -1},
            .withheld = impl_->parse_calls ? impl_->s.markers.tool_call : TokenId{-1}};
}

bool OutputSession::constrained() const noexcept { return impl_ != nullptr && impl_->s.grammar != nullptr; }

void OutputSession::observe_constraint(bool timings, double prepare_seconds) noexcept {
    if (constrained()) { impl_->s.grammar->observe(timings, prepare_seconds); }
}

void OutputSession::constraint_uploaded(std::size_t bytes) noexcept {
    if (constrained()) { impl_->s.grammar->uploaded(bytes); }
}

std::optional<ConstraintObservation> OutputSession::constraint_observation() const {
    if (!constrained()) { return std::nullopt; }
    ConstraintObservation result = impl_->s.grammar->observation();
    const State& state           = impl_->committed;
    result.branch                = state.opened_call || !impl_->tool_calls.empty() ? ConstraintOutputBranch::Tools
                                   : state.opened_content                          ? ConstraintOutputBranch::Content
                                                                                   : ConstraintOutputBranch::Undecided;
    return result;
}

std::uint64_t OutputSession::grammar_masks(std::span<const TokenId> drafts, std::span<std::uint32_t> words) {
    if (!constrained()) { throw std::logic_error("mask requested for unconstrained output"); }
    return impl_->s.grammar->masks(drafts, words);
}

bool OutputSession::constraint_pending() const noexcept { return constrained() && impl_->s.grammar->pending(); }

std::exception_ptr OutputSession::constraint_failure() const noexcept {
    return constrained() ? impl_->s.grammar->failure() : std::exception_ptr{};
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

bool OutputSession::has_tool_calls() const noexcept { return impl_ != nullptr && !impl_->tool_calls.empty(); }

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->committed.diagnostics : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->committed.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{.configured_budget     = impl_->s.thinking.budget,
                               .model_thinking_tokens = impl_->committed.model_thinking_tokens,
                               .injected_tokens       = impl_->committed.injected_tokens,
                               .applied               = impl_->committed.applied};
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->committed.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->committed.matched_stop_order;
    if (index >= impl_->s.policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->s.policy.strings[index].text;
}

} // namespace infernix::models::inkling2
