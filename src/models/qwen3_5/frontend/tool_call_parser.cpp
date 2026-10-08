#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace infernix::models::qwen3_5::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
    // Tolerant mode repaired its opener or kept an undeclared name the strict parser rejects.
    bool repaired = false;
};

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::string_view unquote(std::string_view str) {
    str = trim_format_whitespace(str);
    if (str.size() >= 2) {
        if ((str.front() == '"' && str.back() == '"') ||
            (str.front() == '\'' && str.back() == '\'')) {
            return trim_format_whitespace(str.substr(1, str.size() - 2));
        }
    }
    return str;
}

std::string_view extract_name_from_tag_header(std::string_view header) {
    header = trim_format_whitespace(header);
    if (header.starts_with('=')) {
        return unquote(header.substr(1));
    }
    for (std::size_t i = 0; i < header.size();) {
        if (is_format_whitespace(header[i])) {
            ++i;
            continue;
        }
        const std::size_t attr_begin = i;
        while (i < header.size() && header[i] != '=' && !is_format_whitespace(header[i]) && header[i] != '>') {
            ++i;
        }
        const std::string_view attr_name = header.substr(attr_begin, i - attr_begin);
        skip_format_whitespace(header, i);
        if (i < header.size() && header[i] == '=') {
            ++i;
            skip_format_whitespace(header, i);
            if (i >= header.size()) break;
            std::string_view val;
            if (header[i] == '"' || header[i] == '\'') {
                const char q = header[i];
                const std::size_t q_start = i + 1;
                const std::size_t q_end = header.find(q, q_start);
                if (q_end != std::string_view::npos) {
                    val = header.substr(q_start, q_end - q_start);
                    i = q_end + 1;
                } else {
                    val = header.substr(q_start);
                    i = header.size();
                }
            } else {
                const std::size_t val_begin = i;
                while (i < header.size() && !is_format_whitespace(header[i]) && header[i] != '/' && header[i] != '>') {
                    ++i;
                }
                val = header.substr(val_begin, i - val_begin);
            }
            if (attr_name == "name") {
                return val;
            }
        }
    }
    return unquote(header);
}

bool is_param_open_at(std::string_view text, std::size_t pos, std::size_t& tag_end) {
    std::size_t header_begin = 0;
    if (starts_with_at(text, pos, "<parameter")) {
        header_begin = pos + 10;
    } else if (starts_with_at(text, pos, "<param")) {
        header_begin = pos + 6;
    } else {
        return false;
    }
    if (header_begin >= text.size()) { return false; }
    if (text[header_begin] != '=' && text[header_begin] != ' ' && text[header_begin] != '\t' &&
        text[header_begin] != '\r' && text[header_begin] != '\n' && text[header_begin] != '>') {
        return false;
    }
    const std::size_t end = text.find('>', header_begin);
    if (end != std::string_view::npos && end != header_begin) {
        tag_end = end;
        return true;
    }
    return false;
}

bool is_param_close_at(std::string_view text, std::size_t pos, std::size_t& tag_len) {
    if (starts_with_at(text, pos, "</parameter>")) {
        tag_len = 12;
        return true;
    }
    if (starts_with_at(text, pos, "</param>")) {
        tag_len = 8;
        return true;
    }
    return false;
}

static constexpr std::string_view kToolMarkers[] = {
    "<tool_call>",
    "<function_calls>",
    "<function=",
    "<function ",
    "<function>",
    "<function=\"",
    "<function=\'",
    "<invoke=",
    "<invoke ",
    "<invoke>",
    "<invoke=\"",
    "<invoke=\'",
};

// A constrained tool grammar emits only <tool_call>; the agent-harness forms are recovered from
// free output alone, so constrained content that spells one of them stays content.
std::span<const std::string_view> tool_markers(bool tool_call_only) {
    return tool_call_only ? std::span<const std::string_view>(kToolMarkers, 1)
                          : std::span<const std::string_view>(kToolMarkers);
}

bool is_prefix_of_any_marker(std::string_view prefix, bool tool_call_only) {
    for (const auto& marker : tool_markers(tool_call_only)) {
        if (marker.starts_with(prefix)) { return true; }
    }
    return false;
}

bool matches_any_marker(std::string_view text, bool tool_call_only) {
    for (const auto& marker : tool_markers(tool_call_only)) {
        if (text.starts_with(marker)) { return true; }
    }
    return false;
}

std::size_t find_first_tool_marker(std::string_view text) {
    std::size_t earliest = std::string_view::npos;
    for (const auto& marker : kToolMarkers) {
        std::size_t idx = text.find(marker);
        if (idx != std::string_view::npos && (earliest == std::string_view::npos || idx < earliest)) {
            earliest = idx;
        }
    }
    return earliest;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool tolerant)
        : text_(text), max_name_length_(max_name_length), contract_(contract),
          tolerant_(tolerant) {}

    [[nodiscard]] std::uint32_t duplicate_parameters_repaired() const noexcept {
        return duplicate_parameters_repaired_;
    }

    FallbackReason parse(std::vector<RawToolCall>& calls) {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (starts_with_at(text_, pos, "<tool_call>")) {
                RawToolCall call;
                const FallbackReason terminal = finish_call(calls, call, parse_tool_call(pos, call));
                if (terminal != FallbackReason::None) { return terminal; }
            } else if (starts_with_at(text_, pos, "<function_calls>")) {
                pos += 16;
                bool had_calls = false;
                for (;;) {
                    skip_format_whitespace(text_, pos);
                    if (consume(pos, "</function_calls>")) { break; }
                    if (pos == text_.size()) {
                        // Tolerant: an unclosed wrapper after one or more complete calls is a
                        // truncation; the strict parser keeps the hard structural failure.
                        if (tolerant_ && had_calls) { return FallbackReason::TruncatedTail; }
                        return FallbackReason::MalformedStructure;
                    }
                    RawToolCall call;
                    const FallbackReason terminal = finish_call(calls, call, parse_function(pos, call));
                    if (terminal != FallbackReason::None) { return terminal; }
                    had_calls = true;
                }
                if (!had_calls) { return FallbackReason::MalformedStructure; }
            } else if (starts_with_at(text_, pos, "<function") || starts_with_at(text_, pos, "<invoke")) {
                RawToolCall call;
                const FallbackReason terminal = finish_call(calls, call, parse_function(pos, call));
                if (terminal != FallbackReason::None) { return terminal; }
            } else {
                // Tolerant: a trailing suffix after one or more complete calls is discarded
                // rather than failing the whole output.
                if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    // True when nothing but trailing whitespace remains after `pos` in the tool region.
    bool at_region_end(std::size_t pos) const {
        std::size_t at = pos;
        skip_format_whitespace(text_, at);
        return at == text_.size();
    }

    // Applies tolerant recovery to a sub-parse result and reports whether parse() should
    // terminate. In tolerant mode, a malformed suffix after one or more complete calls is
    // discarded, and a single truncated final call whose name and at least one parameter are
    // complete is retained; the recovered calls are never demoted to text. The strict parser
    // returns the raw failure unchanged.
    FallbackReason finish_call(std::vector<RawToolCall>& calls, RawToolCall& call,
                               FallbackReason failure) {
        if (failure == FallbackReason::None) {
            calls.push_back(std::move(call));
            return FallbackReason::None;
        }
        if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
        if (tolerant_ && failure == FallbackReason::TruncatedTail && calls.empty() &&
            !call.parameters.empty()) {
            calls.push_back(std::move(call));
            return FallbackReason::TruncatedTail;
        }
        return failure;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) {
        if (!consume(pos, "<tool_call>")) { return FallbackReason::MalformedStructure; }
        skip_format_whitespace(text_, pos);
        const FallbackReason failure = parse_function(pos, call);
        if (failure != FallbackReason::None) { return failure; }
        skip_format_whitespace(text_, pos);
        if (consume(pos, "</tool_call>")) { return FallbackReason::None; }
        // Tolerant: a complete call may be followed by explanatory text, or the model may have
        // stopped at the end of its budget before the closing tag. parse() resolves the terminal
        // flag; the strict parser keeps the hard structural failure.
        return tolerant_ ? FallbackReason::TruncatedTail : FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function(std::size_t& pos, RawToolCall& call) {
        std::size_t header_begin = 0;
        std::string_view fn_close = "</function>";
        if (starts_with_at(text_, pos, "<function")) {
            header_begin = pos + 9;
            fn_close = "</function>";
        } else if (starts_with_at(text_, pos, "<invoke")) {
            header_begin = pos + 7;
            fn_close = "</invoke>";
        } else if (tolerant_) {
            // Recover a malformed function opener: a dropped or doubled leading '<', a leaked
            // ChatML turn marker, or a dropped 'function'/'invoke' keyword, followed by '=' or a
            // name. A form that yields no valid name is rejected by valid_function_name below, so
            // prose after a marker cannot pass.
            std::size_t scan = pos;
            while (scan < text_.size() && text_[scan] == '<') { ++scan; }
            if (starts_with_at(text_, scan, "|im_start|>")) { scan += 11; }
            std::size_t kw_len = 0;
            if (starts_with_at(text_, scan, "function")) { kw_len = 8; }
            else if (starts_with_at(text_, scan, "invoke")) { kw_len = 6; }
            else { return FallbackReason::MalformedStructure; }
            scan += kw_len;
            if (scan >= text_.size() || (text_[scan] != '=' && !is_format_whitespace(text_[scan]))) {
                return FallbackReason::MalformedStructure;
            }
            if (text_[scan] == '=') { ++scan; }
            header_begin  = scan;
            fn_close      = kw_len == 8 ? "</function>" : "</invoke>";
            call.repaired = true;
        } else {
            return FallbackReason::MalformedStructure;
        }

        std::size_t tag_end = text_.find('>', header_begin);
        bool ws_boundary = false;
        // Tolerant: the model sometimes drops the '>' after the function name (for example a name
        // followed directly by a newline and a parameter tag). Recover by scanning the identifier
        // run and accepting it when format whitespace separates it from the next '<' or end of
        // region.
        if (tolerant_) {
            std::size_t scan = header_begin;
            while (scan < text_.size() && text_[scan] == '=') { ++scan; }
            const std::size_t ident_begin = scan;
            while (scan < text_.size() && scan - header_begin < max_name_length_) {
                const char byte = text_[scan];
                if (!is_ascii_alphanumeric(byte) && byte != '_' && byte != '-') { break; }
                ++scan;
            }
            if (scan > ident_begin && scan < text_.size() && is_format_whitespace(text_[scan]) &&
                (tag_end == std::string_view::npos || scan < tag_end)) {
                std::size_t after = scan;
                while (after < text_.size() && is_format_whitespace(text_[after])) { ++after; }
                if (after >= text_.size() || text_[after] == '<') {
                    tag_end       = scan;
                    ws_boundary   = true;
                    call.repaired = true;
                }
            }
        }
        if (tag_end == std::string_view::npos || tag_end == header_begin) {
            return FallbackReason::InvalidToolName;
        }
        const std::string_view header = text_.substr(header_begin, tag_end - header_begin);
        call.name = extract_name_from_tag_header(header);
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        // Strict mode rejects a name outside the declared tool set. Tolerant mode keeps an
        // otherwise well-formed call structured and leaves the identity judgment to the consumer:
        // leaking the raw region to content would turn a valid call into prose.
        if (find_tool_contract(contract_, call.name) == nullptr) {
            if (!tolerant_) { return FallbackReason::UndeclaredTool; }
            call.repaired = true;
        }
        pos = ws_boundary ? tag_end : tag_end + 1;

        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, fn_close)) {
                return FallbackReason::None;
            }
            // Tolerant: the region is exhausted after the last complete parameter, so a missing
            // function close is a truncation, not a malformed structure. parse() retains the
            // recovered parameters; the strict parser still requires the closing tag.
            if (tolerant_ && at_region_end(pos)) {
                return FallbackReason::TruncatedTail;
            }
            const FallbackReason failure = parse_parameter(pos, call);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_parameter(std::size_t& pos, RawToolCall& call) {
        std::size_t header_begin = 0;
        std::string_view param_close = "</parameter>";
        if (starts_with_at(text_, pos, "<parameter")) {
            header_begin = pos + 10;
            param_close  = "</parameter>";
        } else if (starts_with_at(text_, pos, "<param")) {
            header_begin = pos + 6;
            param_close  = "</param>";
        } else {
            return FallbackReason::MalformedStructure;
        }

        const std::size_t tag_end = text_.find('>', header_begin);
        if (tag_end == std::string_view::npos || tag_end == header_begin) {
            return FallbackReason::MalformedStructure;
        }
        const std::string_view header = text_.substr(header_begin, tag_end - header_begin);
        const std::string_view name   = extract_name_from_tag_header(header);
        if (name.empty()) {
            return FallbackReason::MalformedStructure;
        }

        const std::size_t value_begin = tag_end + 1;
        std::size_t value_end         = 0;
        std::size_t close_len         = 0;
        if (!find_parameter_close(value_begin, value_end, close_len, param_close)) {
            if (tolerant_) {
                // Tolerant: the region ends before the closing tag, so the output budget cut the
                // parameter value. Keep the value up to the cut (last occurrence wins, as with a
                // complete parameter) and flag the tail; the strict parser keeps the hard
                // structural failure.
                const std::string_view partial = text_.substr(value_begin, text_.size() - value_begin);
                const auto existing = std::find_if(call.parameters.begin(), call.parameters.end(),
                                                   [&](const RawParameter& candidate) { return candidate.name == name; });
                if (existing != call.parameters.end()) {
                    existing->value = partial;
                    ++duplicate_parameters_repaired_;
                } else {
                    call.parameters.push_back(RawParameter{.name = name, .value = partial});
                }
                pos = text_.size();
                return FallbackReason::TruncatedTail;
            }
            return FallbackReason::MalformedStructure;
        }
        const std::string_view value = text_.substr(value_begin, value_end - value_begin);

        const auto existing =
            std::find_if(call.parameters.begin(), call.parameters.end(),
                         [&](const RawParameter& candidate) { return candidate.name == name; });

        // Last occurrence wins, as it would in JSON object syntax, rather than discarding an
        // otherwise well-formed call.
        if (existing != call.parameters.end()) {
            existing->value = value;
            ++duplicate_parameters_repaired_;
        } else {
            call.parameters.push_back(RawParameter{.name = name, .value = value});
        }
        pos = value_end + close_len;
        return FallbackReason::None;
    }

    // After a parameter's closer the grammar continues with another parameter, the function's
    // closer, or (for an output cut short) the region end. A closer followed by anything else is
    // text the value quotes, such as a command that echoes tool markup.
    bool closes_parameter(std::size_t pos) const {
        skip_format_whitespace(text_, pos);
        std::size_t tag_end = 0;
        return pos >= text_.size() || is_param_open_at(text_, pos, tag_end) ||
               starts_with_at(text_, pos, "</function>") || starts_with_at(text_, pos, "</invoke>");
    }

    bool find_parameter_close(std::size_t value_begin, std::size_t& value_end,
                              std::size_t& close_len, std::string_view required_close) const {
        std::size_t depth = 1;
        std::size_t scan  = value_begin;
        while (scan < text_.size()) {
            if (starts_with_at(text_, scan, required_close)) {
                // Closers of parameters the value quotes whole always balance their openers; only
                // the outermost one must be followed by the grammar's next token.
                if (depth != 1 || closes_parameter(scan + required_close.size())) { --depth; }
                if (depth == 0) {
                    value_end = scan;
                    close_len = required_close.size();
                    return true;
                }
                scan += required_close.size();
                continue;
            }
            std::size_t open_tag_end = 0;
            if (is_param_open_at(text_, scan, open_tag_end)) {
                ++depth;
                scan = open_tag_end + 1;
                continue;
            }
            ++scan;
        }
        return false;
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    std::uint32_t duplicate_parameters_repaired_ = 0;
    bool tolerant_ = false;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

// Canonical constrained calls are parsed one at a time. A terminal interruption can discard the
// unfinished suffix while retaining preceding calls. JSON values are delimited lexically, so a
// marker inside a quoted value never becomes a parameter boundary.
class ConstrainedToolRegionParser {
public:
    enum class Status { Complete, Incomplete, Invalid };

    ConstrainedToolRegionParser(std::string_view input, const Contract& contract)
        : input_(input), contract_(contract) {}

    Status parse(std::vector<GeneratedToolCall>& calls, ToolCallParseDiagnostics& diagnostics) {
        while (position_ < input_.size()) {
            if (!calls.empty() && !take("\n")) return status_;
            RawToolCall raw;
            if (!take("<tool_call>\n<function=")) return status_;
            const auto end = input_.find('>', position_);
            if (end == std::string_view::npos) return Status::Incomplete;
            raw.name         = input_.substr(position_, end - position_);
            const auto* tool = find_tool_contract(contract_, raw.name);
            if (!tool) return Status::Invalid;
            position_ = end + 1;
            if (!take("\n")) return status_;
            std::ptrdiff_t previous = -1;
            while (!input_.substr(position_).starts_with("</function>")) {
                if (std::string_view("</function>").starts_with(input_.substr(position_)))
                    return Status::Incomplete;
                if (!take("<parameter=")) return status_;
                const auto name_end = input_.find('>', position_);
                if (name_end == std::string_view::npos) return Status::Incomplete;
                const auto name       = input_.substr(position_, name_end - position_);
                const auto* parameter = find_parameter_contract(*tool, name);
                if (name.empty() || name.find_first_of("<>\r\n") != std::string_view::npos)
                    return Status::Invalid;
                if (tool->strict) {
                    if (!parameter) return Status::Invalid;
                    const auto index = parameter - tool->parameters.data();
                    if (index <= previous) return Status::Invalid;
                    previous = index;
                }
                position_ = name_end + 1;
                if (!take("\n")) return status_;
                const auto value_begin = position_;
                if (parameter && parameter->encoding == Contract::Encoding::Json) {
                    if (!json_value()) return status_;
                } else {
                    const auto close = input_.find("\n</parameter>", position_);
                    if (close == std::string_view::npos) return Status::Incomplete;
                    position_ = close;
                }
                const auto value = input_.substr(value_begin, position_ - value_begin);
                if (!take("\n</parameter>\n")) return status_;
                // Non-strict arguments use the last value for a repeated name, as JSON object
                // consumers do. Keep the first key position and publish each key only once.
                const auto existing =
                    tool->strict
                        ? raw.parameters.end()
                        : std::find_if(raw.parameters.begin(), raw.parameters.end(),
                                       [&](const auto& item) { return item.name == name; });
                if (existing == raw.parameters.end())
                    raw.parameters.push_back({name, value});
                else
                    existing->value = value;
            }
            if (!take("</function>\n</tool_call>")) return status_;
            if (!contract_.parallel && !calls.empty()) return Status::Invalid;
            std::string arguments = "{";
            bool first            = true;
            for (const auto& value : raw.parameters) {
                const auto* parameter = find_parameter_contract(*tool, value.name);
                NormalizedParameter normalized;
                if (parameter && parameter->encoding == Contract::Encoding::RawString) {
                    normalized.json_value = encode_json_string(value.value);
                } else if (parameter && parameter->encoding == Contract::Encoding::Json) {
                    if (!Json::accept(value.value)) return Status::Invalid;
                    normalized.json_value = value.value;
                } else {
                    // The non-strict normalizer owns its framing removal.
                    normalized =
                        normalize_parameter("\n" + std::string(value.value) + "\n", parameter);
                }
                if (normalized.disposition == ParameterNormalization::Omitted) {
                    ++diagnostics.empty_arguments_omitted;
                    continue;
                }
                if (normalized.disposition == ParameterNormalization::SchemaMismatch)
                    ++diagnostics.schema_mismatch_arguments;
                if (!first) arguments += ',';
                first = false;
                arguments += encode_json_string(value.name) + ":" + normalized.json_value;
            }
            arguments += '}';
            calls.push_back(
                {.name = std::string(raw.name), .arguments_json = std::move(arguments)});
        }
        return Status::Complete;
    }

private:
    bool take(std::string_view literal) {
        const auto remaining = input_.substr(position_);
        if (remaining.starts_with(literal)) {
            position_ += literal.size();
            return true;
        }
        status_ = literal.starts_with(remaining) ? Status::Incomplete : Status::Invalid;
        return false;
    }

    bool json_value() {
        bool quoted = false, escaped = false;
        int depth = 0;
        for (; position_ < input_.size(); ++position_) {
            const char c = input_[position_];
            if (quoted) {
                if (escaped)
                    escaped = false;
                else if (c == '\\')
                    escaped = true;
                else if (c == '"')
                    quoted = false;
            } else if (c == '"')
                quoted = true;
            else if (c == '{' || c == '[')
                ++depth;
            else if (c == '}' || c == ']') {
                if (--depth < 0) {
                    status_ = Status::Invalid;
                    return false;
                }
            } else if (c == '\n' && depth == 0)
                return true;
        }
        status_ = Status::Incomplete;
        return false;
    }

    std::string_view input_;
    const Contract& contract_;
    std::size_t position_ = 0;
    Status status_        = Status::Incomplete;
};

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

// Normalizes an accepted region's calls into the structured turn.
void publish_structured_calls(ParsedToolCallOutput& out, const std::vector<RawToolCall>& raw_calls,
                              std::uint32_t duplicate_repairs, const Contract& contract) {
    out.tool_calls.reserve(raw_calls.size());
    for (const RawToolCall& raw : raw_calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
    }
    out.diagnostics.duplicate_parameters_repaired = duplicate_repairs;
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
}

// A turn that ended inside its thinking can hold the calls it meant to make. Accept the earliest
// marker from which the rest of the thinking is nothing but complete, declared calls, parsed
// strictly: prose after a call, a quoted marker or a cut call leaves the thinking as reasoning.
// The prose before the accepted region is never returned.
ParsedToolCallOutput parse_open_reasoning_calls(std::string_view reasoning,
                                                std::size_t max_tool_name_length,
                                                const Contract& contract) {
    std::size_t candidate = find_first_tool_marker(reasoning);
    while (candidate != std::string_view::npos) {
        std::vector<RawToolCall> calls;
        QwenToolRegionParser parser(reasoning.substr(candidate), max_tool_name_length, contract,
                                    false);
        if (parser.parse(calls) == FallbackReason::None) {
            ParsedToolCallOutput out;
            out.diagnostics.marker_seen              = true;
            out.diagnostics.recovered_from_reasoning = true;
            publish_structured_calls(out, calls, parser.duplicate_parameters_repaired(), contract);
            return out;
        }
        const std::size_t next = find_first_tool_marker(reasoning.substr(candidate + 1));
        candidate = next == std::string_view::npos ? next : candidate + 1 + next;
    }
    return {};
}

} // namespace

void MarkdownCodeTracker::feed(std::string_view text) {
    for (const char byte : text) {
        if (byte == '\n') {
            end_line();
        } else {
            line_.push_back(byte);
        }
    }
}

void MarkdownCodeTracker::end_line() {
    std::string_view line(line_);
    const std::size_t first = line.find_first_not_of(" \t\r\f\v");
    line = first == std::string_view::npos ? std::string_view{} : line.substr(first);
    const auto run_of_three = [](std::string_view text, char fence) {
        return text.size() >= 3 && text[0] == fence && text[1] == fence && text[2] == fence;
    };
    if (fence_ != '\0') {
        if (run_of_three(line, fence_)) { fence_ = '\0'; }
    } else if (!line.empty() && (line[0] == '`' || line[0] == '~') && run_of_three(line, line[0]) &&
               line.substr(3).find(std::string(3, line[0])) == std::string_view::npos) {
        // A ``` or ~~~ opener whose run is not closed on the same line (that is inline code).
        fence_ = line[0];
    } else if (line.empty()) {
        ticks_ = 0; // a blank line ends the paragraph and any inline code left open in it
    } else {
        ticks_ += static_cast<std::uint32_t>(std::count(line_.begin(), line_.end(), '`'));
    }
    line_.clear();
}

bool MarkdownCodeTracker::in_code() const noexcept {
    return fence_ != '\0' ||
           (ticks_ + static_cast<std::uint32_t>(std::count(line_.begin(), line_.end(), '`'))) % 2 == 1;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract,
                                                 bool tolerant, MarkdownCodeTracker code) {
    const std::string_view source(text);
    // A marker of free output inside Markdown code is a quoted example, never a call. Candidates
    // only move forward, so the tracker reads each byte once.
    std::size_t tracked = 0;
    const auto quoted   = [&](std::size_t at) {
        if (contract.constrained) { return false; }
        code.feed(source.substr(tracked, at - tracked));
        tracked = at;
        return code.in_code();
    };
    const auto next_marker = [&](std::size_t from) {
        const std::size_t found = find_first_tool_marker(source.substr(from));
        return found == std::string_view::npos ? std::string::npos : from + found;
    };
    std::size_t candidate = find_first_tool_marker(source);
    while (candidate != std::string::npos && quoted(candidate)) { candidate = next_marker(candidate + 1); }
    if (candidate == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try the first marker, then
    // each later `<tool_call>` wrapper, and accept the first region that parses; earlier markers
    // stay ordinary content. A truncated tail that still kept a complete call (tolerant mode) is a
    // recovered region, not a failure.
    std::vector<RawToolCall> raw_calls;
    std::size_t accepted                   = std::string::npos;
    FallbackReason accepted_reason         = FallbackReason::None;
    std::uint32_t duplicate_repairs        = 0;
    FallbackReason first_failure           = FallbackReason::MalformedStructure;
    bool first_failure_recorded            = false;
    while (candidate != std::string::npos) {
        std::vector<RawToolCall> calls;
        QwenToolRegionParser parser(source.substr(candidate), max_tool_name_length, contract,
                                    tolerant);
        const FallbackReason failure = parser.parse(calls);
        if (failure == FallbackReason::None ||
            (failure == FallbackReason::TruncatedTail && !calls.empty())) {
            accepted          = candidate;
            accepted_reason   = failure;
            duplicate_repairs = parser.duplicate_parameters_repaired();
            raw_calls         = std::move(calls);
            break;
        }
        if (!first_failure_recorded) {
            first_failure          = failure;
            first_failure_recorded = true;
        }
        // Retries move only to a later `<tool_call>` wrapper: the markup nested inside a failed
        // region (its `<function=...>` or `<invoke>`) must not re-read a truncated call.
        candidate = text.find(kToolOpen, candidate + 1);
        while (candidate != std::string::npos && quoted(candidate)) {
            candidate = text.find(kToolOpen, candidate + 1);
        }
    }
    if (accepted == std::string::npos) {
        // No region parsed. A truncated tail that kept no call carries no arguments either, so
        // the response is returned as text with the first region's reason recorded.
        out.diagnostics.fallback_reason = first_failure;
        return fallback(text, out.diagnostics);
    }
    // A recovered truncated tail keeps its reason for transparency without demoting the output.
    // Only tolerant mode produces one, or repairs a kept call.
    out.diagnostics.fallback_reason = accepted_reason;
    out.diagnostics.tolerant_recovered =
        accepted_reason == FallbackReason::TruncatedTail ||
        std::any_of(raw_calls.begin(), raw_calls.end(),
                    [](const RawToolCall& call) { return call.repaired; });

    out.content = rtrim_format_whitespace(source.substr(0, accepted));
    publish_structured_calls(out, raw_calls, duplicate_repairs, contract);
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length, bool tolerant)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length),
      tolerant_(tolerant) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    fed_content_ = true;
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    const bool tool_call_only = contract_->constrained;
    std::string visible;
    // Every published byte also feeds the code tracker; the held whitespace joins it when it is
    // published, so the tracker always describes the text before the held bytes.
    const auto publish = [&](std::string_view bytes) {
        visible.append(bytes);
        code_.feed(bytes);
    };
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (!pending_tag_.empty()) {
            pending_tag_.push_back(byte);
            if (matches_any_marker(pending_tag_, tool_call_only)) {
                MarkdownCodeTracker before_marker = code_;
                before_marker.feed(trailing_whitespace_);
                if (!contract_->constrained && before_marker.in_code()) {
                    // A quoted example inside Markdown code: the marker is text.
                    publish(trailing_whitespace_);
                    trailing_whitespace_.clear();
                    publish(pending_tag_);
                    pending_tag_.clear();
                    continue;
                }
                region_code_ = code_;
                tool_region_ = std::move(trailing_whitespace_);
                trailing_whitespace_.clear();
                tool_region_.append(pending_tag_);
                pending_tag_.clear();
                tool_region_.append(text.substr(index + 1));
                saw_tool_marker_ = true;
                break;
            }
            if (is_prefix_of_any_marker(pending_tag_, tool_call_only)) { continue; }
            // Publish the bytes before this one; the byte itself may start the next marker.
            pending_tag_.pop_back();
            publish(trailing_whitespace_);
            trailing_whitespace_.clear();
            publish(pending_tag_);
            pending_tag_.clear();
        }

        if (byte == '<') {
            pending_tag_.push_back(byte);
        } else if (is_format_whitespace(byte)) {
            trailing_whitespace_.push_back(byte);
        } else {
            publish(trailing_whitespace_);
            trailing_whitespace_.clear();
            publish(std::string_view(&byte, 1));
        }
    }
    const auto suppressed = std::min(continuation_withheld_bytes_, visible.size());
    visible.erase(0, suppressed);
    continuation_withheld_bytes_ -= suppressed;
    return visible;
}

void ToolCallOutputDecoder::initialize_continuation(std::string_view prefix) {
    (void)feed(prefix);
    if (contract_ && contract_->constrained && saw_tool_marker_) {
        std::vector<GeneratedToolCall> calls;
        ToolCallParseDiagnostics diagnostics;
        auto region = std::string_view(tool_region_);
        region.remove_prefix(region.find(kToolOpen));
        const auto status =
            ConstrainedToolRegionParser(region, *contract_).parse(calls, diagnostics);
        if (!calls.empty() || status == ConstrainedToolRegionParser::Status::Invalid)
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed or malformed calls");
    } else if (saw_tool_marker_) {
        std::vector<RawToolCall> calls;
        (void)QwenToolRegionParser(tool_region_, max_tool_name_length_, *contract_, tolerant_)
            .parse(calls);
        if (!calls.empty())
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed calls");
    }
    continuation_withheld_bytes_ =
        trailing_whitespace_.size() + pending_tag_.size() + tool_region_.size();
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish(FinishReason reason,
                                                              std::string_view open_reasoning) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    // Only a turn the model ended itself recovers calls from its open thinking; an output,
    // context or cancellation cut leaves a call it was still deliberating over unexecuted.
    if (tolerant_ && !contract_->constrained && reason == FinishReason::StopToken &&
        !open_reasoning.empty() && !fed_content_) {
        ParsedToolCallOutput recovered =
            parse_open_reasoning_calls(open_reasoning, max_tool_name_length_, *contract_);
        if (recovered.is_tool_call_response) {
            return Terminal{.content     = {},
                            .tool_calls  = std::move(recovered.tool_calls),
                            .diagnostics = recovered.diagnostics};
        }
    }

    if (contract_->constrained && saw_tool_marker_) {
        Terminal result;
        result.diagnostics.marker_seen = true;
        auto region                    = std::string_view(tool_region_);
        region.remove_prefix(region.find(kToolOpen));
        const auto status = ConstrainedToolRegionParser(region, *contract_)
                                .parse(result.tool_calls, result.diagnostics);
        const bool interrupted = reason == FinishReason::OutputLimit ||
                                 reason == FinishReason::ContextCapacity ||
                                 reason == FinishReason::Cancelled;
        if (status == ConstrainedToolRegionParser::Status::Invalid ||
            (!interrupted && status != ConstrainedToolRegionParser::Status::Complete))
            throw std::logic_error("committed tool grammar and parameter decoder disagree");
        result.diagnostics.structured_call_count =
            static_cast<std::uint32_t>(result.tool_calls.size());
        return result;
    }
    if (contract_->constrained && contract_->required && reason == FinishReason::StopToken)
        throw std::logic_error("required tool grammar ended without a call");
    if (contract_->constrained && contract_->required) return {};

    ParsedToolCallOutput parsed = parse_qwen_tool_call_output(tool_region_, max_tool_name_length_,
                                                              *contract_, tolerant_, region_code_);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        // The parser reports the held bytes before the accepted structured region, which are the
        // bytes after an earlier quoted marker that this decoder has not published yet.
        std::string content = std::move(parsed.content);
        trailing_whitespace_.clear();
        tool_region_.clear();
        pending_tag_.clear();
        return Terminal{.content     = std::move(content),
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }

    std::string tail = std::move(trailing_whitespace_);
    tail.append(pending_tag_);
    pending_tag_.clear();
    tail += tool_region_;
    tail.erase(0, std::min(continuation_withheld_bytes_, tail.size()));
    continuation_withheld_bytes_ = 0;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

} // namespace infernix::models::qwen3_5::frontend
