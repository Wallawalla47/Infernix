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

// The wrapper-less openers agent harnesses drift into. They have no token of their own, so free
// output reads them by their spelling; a constrained grammar emits only the `<tool_call>` token,
// so constrained content that spells one of them stays content.
static constexpr std::string_view kBareToolMarkers[] = {
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

bool is_prefix_of_bare_marker(std::string_view prefix) {
    for (const auto& marker : kBareToolMarkers) {
        if (marker.starts_with(prefix)) { return true; }
    }
    return false;
}

bool matches_bare_marker(std::string_view text) {
    for (const auto& marker : kBareToolMarkers) {
        if (text.starts_with(marker)) { return true; }
    }
    return false;
}

// The first bare marker at or after `from`, or npos.
std::size_t find_bare_marker(std::string_view text, std::size_t from) {
    std::size_t earliest = std::string_view::npos;
    for (const auto& marker : kBareToolMarkers) {
        const std::size_t found = text.find(marker, from);
        if (found < earliest) { earliest = found; }
    }
    return earliest;
}

// The first `<tool_call>` token at or after `from`, or npos. Offsets are ascending.
std::size_t next_tool_call_token(std::span<const std::size_t> tokens, std::size_t from) {
    const auto found = std::lower_bound(tokens.begin(), tokens.end(), from);
    return found == tokens.end() ? std::string_view::npos : *found;
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

// Model bytes need not be valid UTF-8 (a recovered call can end inside a sequence): invalid
// sequences encode as U+FFFD instead of throwing.
std::string encode_json_string(std::string_view value) {
    return Json(std::string(value)).dump(-1, ' ', false, Json::error_handler_t::replace);
}

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types,
                                                bool nullable = false) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String) || nullable) {
        // A string that may be null is raw text: `null` is the null value (BX-06), anything else text.
        if ((nullable || admits_type(types, SchemaType::Null)) && trim_format_whitespace(framed) == "null") {
            return {.json_value = "null"};
        }
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
    if (parameter != nullptr && (parameter->policy == NormalizationPolicy::DeclaredTypes || parameter->nullable)) {
        return normalize_declared_parameter(encoded_value, parameter->types, parameter->nullable);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

// Parses one region of generated text. `tool_calls` are the `<tool_call>` token offsets of the
// whole text the region starts at `base` of.
class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::span<const std::size_t> tool_calls,
                         std::size_t base, std::size_t max_name_length, const Contract& contract,
                         bool tolerant)
        : text_(text), tool_calls_(tool_calls), base_(base), max_name_length_(max_name_length),
          contract_(contract), tolerant_(tolerant) {}

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
            if (tool_call_at(pos)) {
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

    // Whether the model's `<tool_call>` token starts at `pos`; its spelling alone opens nothing.
    bool tool_call_at(std::size_t pos) const {
        return std::binary_search(tool_calls_.begin(), tool_calls_.end(), base_ + pos);
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
        if (!tool_call_at(pos)) { return FallbackReason::MalformedStructure; }
        pos += kToolCallOpen.size();
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
        // A drifted opener that names no keyword (`<=NAME>`, `<function:NAME>`, a bare `<NAME>`)
        // is a call only for a declared tool; otherwise the region stays text.
        bool declared_only = false;
        // A bare `<NAME>` may also close as `</NAME>`.
        bool bare_name     = false;
        // The keyword ends where a name could not continue: `<function_lookup>` names a tool.
        const auto keyword_at = [&](std::size_t at, std::string_view keyword) {
            if (!starts_with_at(text_, at, keyword)) { return false; }
            const std::size_t next = at + keyword.size();
            return next >= text_.size() ||
                   (!is_ascii_alphanumeric(text_[next]) && text_[next] != '_' && text_[next] != '-');
        };
        // `<function:NAME>` is a drifted opener, not the keyword's header.
        const auto opener_at = [&](std::string_view keyword) {
            const std::size_t next = pos + keyword.size();
            return keyword_at(pos, keyword) && (next >= text_.size() || text_[next] != ':');
        };
        if (opener_at("<function")) {
            // `<function =NAME>` needs no repair: the header's leading whitespace is trimmed.
            header_begin = pos + 9;
            fn_close = "</function>";
        } else if (opener_at("<invoke")) {
            header_begin = pos + 7;
            fn_close = "</invoke>";
        } else if (tolerant_) {
            // Recover a malformed function opener after any run of leading '<' (dropped or
            // doubled) and a leaked ChatML turn marker:
            // - the 'function'/'invoke' keyword followed by '=', whitespace or ':' (`:` only for a
            //   declared name);
            // - no keyword: '=' and the name (`<=NAME>`), or the bare name (`<NAME>`), both only
            //   for a declared name and only after a '<' or the turn marker.
            // A form that yields no valid name is rejected by valid_function_name below, so prose
            // after a marker cannot pass.
            std::size_t scan = pos;
            while (scan < text_.size() && text_[scan] == '<') { ++scan; }
            if (starts_with_at(text_, scan, "|im_start|>")) { scan += 11; }
            const bool opened = scan != pos;
            std::size_t kw_len = 0;
            if (keyword_at(scan, "function")) { kw_len = 8; }
            else if (keyword_at(scan, "invoke")) { kw_len = 6; }
            const std::size_t after_kw = scan + kw_len;
            if (kw_len != 0 && after_kw < text_.size() &&
                (text_[after_kw] == '=' || text_[after_kw] == ':' ||
                 is_format_whitespace(text_[after_kw]))) {
                declared_only = text_[after_kw] == ':';
                header_begin  = text_[after_kw] == ':' || text_[after_kw] == '=' ? after_kw + 1
                                                                                 : after_kw;
                fn_close      = kw_len == 8 ? "</function>" : "</invoke>";
            } else if (opened && scan < text_.size() && text_[scan] == '=') {
                declared_only = true;
                header_begin  = scan + 1;
            } else if (opened && scan < text_.size() &&
                       (is_ascii_alphanumeric(text_[scan]) || text_[scan] == '_')) {
                declared_only = true;
                bare_name     = true;
                header_begin  = scan;
            } else {
                return FallbackReason::MalformedStructure;
            }
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
        // A drifted opener without its keyword is too weak a signal for that: it is a call only for
        // a declared name.
        current_tool_ = find_tool_contract(contract_, call.name);
        if (current_tool_ == nullptr) {
            if (!tolerant_) { return FallbackReason::UndeclaredTool; }
            if (declared_only) { return FallbackReason::MalformedStructure; }
            call.repaired = true;
        }
        pos         = ws_boundary ? tag_end : tag_end + 1;
        bare_close_ = bare_name ? "</" + std::string(call.name) + ">" : std::string();

        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, fn_close)) {
                return FallbackReason::None;
            }
            if (bare_name && starts_with_at(text_, pos, "</") &&
                starts_with_at(text_, pos + 2, call.name) &&
                starts_with_at(text_, pos + 2 + call.name.size(), ">")) {
                pos += call.name.size() + 3;
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
        } else if (std::size_t damaged_end = 0; damaged_param_open_at(pos, damaged_end)) {
            // Tolerant: `<=NAME>` reads as `<parameter=NAME>` (its header is "=NAME").
            header_begin  = pos + 1;
            param_close   = "</parameter>";
            call.repaired = true;
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
        return pos >= text_.size() || is_param_open_at(text_, pos, tag_end) || damaged_param_open_at(pos, tag_end) ||
               starts_with_at(text_, pos, "</function>") || starts_with_at(text_, pos, "</invoke>") ||
               (!bare_close_.empty() && starts_with_at(text_, pos, bare_close_));
    }

    // Tolerant: a parameter opener that lost its keyword, `<=NAME>`, for a parameter the call's tool
    // declares (Strata PR #3960: drifting low-bit quants write it after a `</parameter>`). Without
    // the repair the previous `</parameter>` is followed by no grammar token, so the next parameter
    // ends up inside the previous value. A name the tool does not declare stays value text.
    bool damaged_param_open_at(std::size_t pos, std::size_t& tag_end) const {
        if (!tolerant_ || current_tool_ == nullptr || !starts_with_at(text_, pos, "<=")) { return false; }
        const std::size_t limit = std::min(text_.size(), pos + 2 + max_name_length_ + 1);
        std::size_t end         = pos + 2;
        while (end < limit && text_[end] != '>' && text_[end] != '<' && text_[end] != '\n') { ++end; }
        if (end >= limit || text_[end] != '>' || end == pos + 2) { return false; }
        const std::string_view name = extract_name_from_tag_header(text_.substr(pos + 1, end - pos - 1));
        if (name.empty() || find_parameter_contract(*current_tool_, name) == nullptr) { return false; }
        tag_end = end;
        return true;
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
    std::span<const std::size_t> tool_calls_;
    std::size_t base_ = 0;
    std::size_t max_name_length_;
    const Contract& contract_;
    std::uint32_t duplicate_parameters_repaired_ = 0;
    bool tolerant_ = false;
    // The current call's `</NAME>` when it opened as a bare `<NAME>` (tolerant), else empty.
    std::string bare_close_;
    // The current call's declared tool, or null for an undeclared name.
    const Contract::Tool* current_tool_ = nullptr;
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

ParsedToolCallOutput fallback(std::string_view text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = std::string(text);
    out.diagnostics = diagnostics;
    return out;
}

// A bare opener right after a `<tool_call>` spelled with ordinary tokens (only format whitespace
// between) belongs to that spelled wrapper, which is text.
bool after_spelled_wrapper(GeneratedText output, std::size_t at) {
    std::size_t end = at;
    while (end != 0 && is_format_whitespace(output.text[end - 1])) { --end; }
    if (end < kToolCallOpen.size()) { return false; }
    const std::size_t wrapper = end - kToolCallOpen.size();
    return output.text.substr(wrapper, kToolCallOpen.size()) == kToolCallOpen &&
           !std::binary_search(output.tool_call_tokens.begin(), output.tool_call_tokens.end(),
                               wrapper);
}

// The first call opener at or after `from`: a `<tool_call>` token, or for free output a bare
// marker spelled in the text.
std::size_t next_call_marker(GeneratedText output, const Contract& contract, std::size_t from) {
    const std::size_t token = next_tool_call_token(output.tool_call_tokens, from);
    if (contract.constrained) { return token; }
    std::size_t bare = find_bare_marker(output.text, from);
    while (bare < token && after_spelled_wrapper(output, bare)) {
        bare = find_bare_marker(output.text, bare + 1);
    }
    return std::min(token, bare);
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
ParsedToolCallOutput parse_open_reasoning_calls(GeneratedText reasoning,
                                                std::size_t max_tool_name_length,
                                                const Contract& contract) {
    std::size_t candidate = next_call_marker(reasoning, contract, 0);
    while (candidate != std::string_view::npos) {
        std::vector<RawToolCall> calls;
        QwenToolRegionParser parser(reasoning.text.substr(candidate), reasoning.tool_call_tokens,
                                    candidate, max_tool_name_length, contract, false);
        if (parser.parse(calls) == FallbackReason::None) {
            ParsedToolCallOutput out;
            out.diagnostics.marker_seen              = true;
            out.diagnostics.recovered_from_reasoning = true;
            publish_structured_calls(out, calls, parser.duplicate_parameters_repaired(), contract);
            return out;
        }
        candidate = next_call_marker(reasoning, contract, candidate + 1);
    }
    return {};
}

constexpr std::string_view kFirstOpener   = "<tool_call>\n<function=";
constexpr std::string_view kNextOpener    = "\n<tool_call>\n<function=";
constexpr std::string_view kParameterOpen = "<parameter=";
constexpr std::string_view kValueEnd      = "\n</parameter>";
constexpr std::string_view kJsonValueEnd  = "</parameter>\n";
constexpr std::string_view kCallClose     = "</function>\n</tool_call>";

// A JSON string's characters without its quotes. Valid UTF-8 escapes character by character, so
// the pieces of a value concatenate to the encoding of the whole.
void append_json_string_body(std::string& out, std::string_view text) {
    const std::string quoted = encode_json_string(text);
    out.append(quoted, 1, quoted.size() - 2);
}

} // namespace

// The constrained route's calls, read once as the grammar emits them (its canonical framing, see
// tool_grammar.cpp): `<tool_call>\n<function=NAME>\n`, `<parameter=NAME>\nVALUE\n</parameter>\n` per
// parameter, `</function>\n</tool_call>`, and `\n` before the next call. Each call's arguments JSON is
// published as it is generated, byte for byte the arguments_json of the complete call. A string value
// streams as its JSON string and a strict JSON value verbatim; a value whose JSON depends on all of it
// (a non-strict value of a declared non-string type, or of no declared type) is published when it
// closes. The grammar's companion names each parameter once per call, so a published key never
// repeats. Only complete calls are returned; an interrupted turn leaves the last call unfinished.
class ConstrainedToolStream {
public:
    enum class Status { Complete, Incomplete, Invalid };

    explicit ConstrainedToolStream(const Contract& contract) : contract_(contract) {}

    // The next region bytes; the region starts at its first `<tool_call>` token.
    void feed(std::string_view bytes) {
        for (const char byte : bytes) {
            if (invalid_) { return; }
            step(byte);
        }
        flush_string();
    }

    [[nodiscard]] Status status() const noexcept {
        if (invalid_) { return Status::Invalid; }
        return phase_ == Phase::Opener && matched_ == 0 && !calls_.empty() ? Status::Complete : Status::Incomplete;
    }

    [[nodiscard]] std::vector<GeneratedToolCall>& calls() noexcept { return calls_; }
    [[nodiscard]] const ToolCallParseDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    [[nodiscard]] std::vector<ToolCallDelta> take_deltas() noexcept { return std::exchange(deltas_, {}); }

private:
    enum class Phase : std::uint8_t {
        Opener,
        FunctionName,
        FunctionEnd,    // the `\n` after the name's `>`
        Parameters,     // `<`, then `p` opens a parameter and `/` closes the call
        ParameterOpen,
        CallClose,
        ParameterName,
        ValueStart,     // the `\n` after the name's `>`
        Value,
        ValueClose,     // what follows the value's terminator
    };
    enum class ValueMode : std::uint8_t { String, Json, Buffered };

    // Matches the next byte of `text`; true once all of it has matched.
    bool match(std::string_view text, char byte) {
        if (byte != text[matched_]) {
            invalid_ = true;
            return false;
        }
        if (++matched_ != text.size()) { return false; }
        matched_ = 0;
        return true;
    }

    void step(char byte) {
        switch (phase_) {
        case Phase::Opener:
            if (!calls_.empty() && !contract_.parallel) {
                invalid_ = true;
            } else if (match(calls_.empty() ? kFirstOpener : kNextOpener, byte)) {
                phase_ = Phase::FunctionName;
                name_.clear();
            }
            return;
        case Phase::FunctionName:
            if (byte != '>') {
                name_.push_back(byte);
                return;
            }
            tool_ = find_tool_contract(contract_, name_);
            if (tool_ == nullptr) {
                invalid_ = true;
                return;
            }
            start_call();
            phase_ = Phase::FunctionEnd;
            return;
        case Phase::FunctionEnd:
        case Phase::ValueStart:
            if (byte != '\n') {
                invalid_ = true;
            } else if (phase_ == Phase::FunctionEnd) {
                phase_ = Phase::Parameters;
            } else {
                begin_value();
            }
            return;
        case Phase::Parameters:
            if (matched_ == 0) {
                if (byte == '<') {
                    matched_ = 1;
                } else {
                    invalid_ = true;
                }
            } else if (byte == kParameterOpen[1]) {
                phase_   = Phase::ParameterOpen;
                matched_ = 2;
            } else if (byte == kCallClose[1]) {
                phase_   = Phase::CallClose;
                matched_ = 2;
            } else {
                invalid_ = true;
            }
            return;
        case Phase::ParameterOpen:
            if (match(kParameterOpen, byte)) {
                phase_ = Phase::ParameterName;
                name_.clear();
            }
            return;
        case Phase::CallClose:
            if (match(kCallClose, byte)) { complete_call(); }
            return;
        case Phase::ParameterName:
            if (byte == '>') {
                end_parameter_name();
            } else if (byte == '<' || byte == '\r' || byte == '\n') {
                invalid_ = true;
            } else {
                name_.push_back(byte);
            }
            return;
        case Phase::Value: value_byte(byte); return;
        case Phase::ValueClose:
            if (match(mode_ == ValueMode::Json ? kJsonValueEnd : std::string_view("\n"), byte)) {
                phase_ = Phase::Parameters;
            }
            return;
        }
    }

    void start_call() {
        call_        = GeneratedToolCall{.name = name_, .arguments_json = {}};
        names_.clear();
        previous_    = -1;
        first_pair_  = true;
        omitted_     = 0;
        mismatched_  = 0;
        deltas_.push_back(ToolCallDelta{.index = static_cast<std::uint32_t>(calls_.size()), .name = name_});
        emit("{");
    }

    void complete_call() {
        emit("}");
        deltas_.back().complete = true;
        calls_.push_back(std::move(call_));
        diagnostics_.empty_arguments_omitted += omitted_;
        diagnostics_.schema_mismatch_arguments += mismatched_;
        diagnostics_.structured_call_count = static_cast<std::uint32_t>(calls_.size());
        phase_ = Phase::Opener;
    }

    void end_parameter_name() {
        if (name_.empty() || std::find(names_.begin(), names_.end(), name_) != names_.end()) {
            invalid_ = true;
            return;
        }
        parameter_ = find_parameter_contract(*tool_, name_);
        if (tool_->strict) {
            if (parameter_ == nullptr) {
                invalid_ = true;
                return;
            }
            const auto index = parameter_ - tool_->parameters.data();
            if (index <= previous_) {
                invalid_ = true;
                return;
            }
            previous_ = index;
        }
        names_.push_back(name_);
        phase_ = Phase::ValueStart;
    }

    // `"name":` with its separating comma.
    std::string pair_prefix() {
        std::string prefix = first_pair_ ? "" : ",";
        first_pair_        = false;
        prefix += encode_json_string(name_);
        prefix.push_back(':');
        return prefix;
    }

    void begin_value() {
        using Encoding = Contract::Encoding;
        // A string that may be null is held until it ends: `null` is the null value, not text (BX-06).
        const bool nullable_string =
            parameter_ != nullptr &&
            (parameter_->nullable || (parameter_->policy == NormalizationPolicy::DeclaredTypes &&
                                      admits_type(parameter_->types, SchemaType::String) &&
                                      admits_type(parameter_->types, SchemaType::Null)));
        if (parameter_ != nullptr && parameter_->encoding == Encoding::Json) {
            mode_ = ValueMode::Json;
        } else if (nullable_string) {
            mode_ = ValueMode::Buffered;
        } else if (parameter_ != nullptr &&
                   (parameter_->encoding == Encoding::RawString ||
                    (parameter_->policy == NormalizationPolicy::DeclaredTypes &&
                     admits_type(parameter_->types, SchemaType::String)))) {
            // A string-admitting declared type keeps the value text as a string (the canonical
            // framing is exactly one line break on each side).
            mode_ = ValueMode::String;
        } else {
            mode_ = ValueMode::Buffered;
        }
        value_.clear();
        held_    = 0;
        quoted_  = false;
        escaped_ = false;
        depth_   = 0;
        if (mode_ == ValueMode::Json) { emit(pair_prefix()); }
        if (mode_ == ValueMode::String) { emit(pair_prefix() + "\""); }
        phase_ = Phase::Value;
    }

    void value_byte(char byte) {
        if (mode_ == ValueMode::Json) {
            // Delimited lexically: a `\n` outside strings at depth 0 ends the value.
            if (quoted_) {
                if (escaped_) {
                    escaped_ = false;
                } else if (byte == '\\') {
                    escaped_ = true;
                } else if (byte == '"') {
                    quoted_ = false;
                }
            } else if (byte == '"') {
                quoted_ = true;
            } else if (byte == '{' || byte == '[') {
                ++depth_;
            } else if (byte == '}' || byte == ']') {
                if (--depth_ < 0) {
                    invalid_ = true;
                    return;
                }
            } else if (byte == '\n' && depth_ == 0) {
                if (!Json::accept(value_)) {
                    invalid_ = true;
                    return;
                }
                phase_ = Phase::ValueClose;
                return;
            }
            value_.push_back(byte);
            emit(std::string_view(&byte, 1));
            return;
        }
        // A raw value ends at its first `\n</parameter>`; `\n` occurs in it only at its start.
        if (byte == kValueEnd[held_]) {
            if (++held_ == kValueEnd.size()) {
                held_ = 0;
                end_raw_value();
            }
            return;
        }
        value_.append(kValueEnd.substr(0, held_));
        held_ = 0;
        if (byte == kValueEnd[0]) {
            held_ = 1;
        } else {
            value_.push_back(byte);
        }
    }

    void end_raw_value() {
        if (mode_ == ValueMode::String) {
            flush_string(true);
            emit("\"");
        } else {
            const NormalizedParameter normalized = normalize_parameter("\n" + value_ + "\n", parameter_);
            if (normalized.disposition == ParameterNormalization::Omitted) {
                ++omitted_;
            } else {
                if (normalized.disposition == ParameterNormalization::SchemaMismatch) { ++mismatched_; }
                emit(pair_prefix() + normalized.json_value);
            }
        }
        value_.clear();
        phase_ = Phase::ValueClose;
    }

    // Publishes a string value's bytes read so far, short of a possible terminator.
    // A feed may end inside a code point (a token can carry part of one), so a trailing incomplete
    // UTF-8 sequence waits for its continuation bytes unless the value has ended (`all`).
    void flush_string(bool all = false) {
        if (phase_ != Phase::Value || mode_ != ValueMode::String || value_.empty()) { return; }
        std::size_t complete = value_.size();
        if (!all) {
            std::size_t lead = value_.size();
            while (lead > 0 && value_.size() - lead < 4 &&
                   (static_cast<unsigned char>(value_[lead - 1]) & 0xC0U) == 0x80U) {
                --lead;
            }
            if (lead > 0) {
                const auto byte        = static_cast<unsigned char>(value_[lead - 1]);
                const std::size_t need = byte >= 0xF0U ? 4 : byte >= 0xE0U ? 3 : byte >= 0xC0U ? 2 : 1;
                if (value_.size() - (lead - 1) < need) { complete = lead - 1; }
            }
        }
        if (complete == 0) { return; }
        std::string escaped;
        append_json_string_body(escaped, std::string_view(value_).substr(0, complete));
        value_.erase(0, complete);
        emit(escaped);
    }

    void emit(std::string_view text) {
        if (text.empty()) { return; }
        call_.arguments_json.append(text);
        const auto index = static_cast<std::uint32_t>(calls_.size());
        if (deltas_.empty() || deltas_.back().index != index || deltas_.back().complete) {
            deltas_.push_back(ToolCallDelta{.index = index});
        }
        deltas_.back().arguments.append(text);
    }

    const Contract& contract_;
    Phase phase_         = Phase::Opener;
    std::size_t matched_ = 0;
    bool invalid_        = false;
    std::string name_; // the function or parameter name being read
    const Contract::Tool* tool_           = nullptr;
    const Contract::Parameter* parameter_ = nullptr;
    std::ptrdiff_t previous_              = -1; // a strict call's last parameter index
    std::vector<std::string> names_;            // the current call's parameter names
    GeneratedToolCall call_;                    // the current call, its arguments so far
    bool first_pair_         = true;
    std::uint32_t omitted_    = 0;
    std::uint32_t mismatched_ = 0;
    ValueMode mode_          = ValueMode::String;
    // A JSON or buffered value; a string value's bytes not yet published.
    std::string value_;
    std::size_t held_ = 0; // a raw value's bytes matching its terminator so far
    bool quoted_      = false;
    bool escaped_     = false;
    int depth_        = 0;
    std::vector<GeneratedToolCall> calls_;
    std::vector<ToolCallDelta> deltas_;
    ToolCallParseDiagnostics diagnostics_;
};

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

ParsedToolCallOutput parse_qwen_tool_call_output(GeneratedText output,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract,
                                                 bool tolerant, MarkdownCodeTracker code) {
    const std::string_view source = output.text;
    // A marker of free output inside Markdown code is a quoted example, never a call. Candidates
    // only move forward, so the tracker reads each byte once.
    std::size_t tracked = 0;
    const auto quoted   = [&](std::size_t at) {
        if (contract.constrained) { return false; }
        code.feed(source.substr(tracked, at - tracked));
        tracked = at;
        return code.in_code();
    };
    std::size_t candidate = next_call_marker(output, contract, 0);
    while (candidate != std::string::npos && quoted(candidate)) {
        candidate = next_call_marker(output, contract, candidate + 1);
    }
    if (candidate == std::string::npos) { return fallback(source); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try the first marker, then
    // each later `<tool_call>` token, and accept the first region that parses; earlier markers
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
        QwenToolRegionParser parser(source.substr(candidate), output.tool_call_tokens, candidate,
                                    max_tool_name_length, contract, tolerant);
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
        // Retries move only to a later `<tool_call>` token: the markup nested inside a failed
        // region (its `<function=...>` or `<invoke>`) must not re-read a truncated call.
        candidate = next_tool_call_token(output.tool_call_tokens, candidate + 1);
        while (candidate != std::string::npos && quoted(candidate)) {
            candidate = next_tool_call_token(output.tool_call_tokens, candidate + 1);
        }
    }
    if (accepted == std::string::npos) {
        // No region parsed. A truncated tail that kept no call carries no arguments either, so
        // the response is returned as text with the first region's reason recorded.
        out.diagnostics.fallback_reason = first_failure;
        return fallback(source, out.diagnostics);
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

ToolCallOutputDecoder::~ToolCallOutputDecoder()                                          = default;
ToolCallOutputDecoder::ToolCallOutputDecoder(ToolCallOutputDecoder&&) noexcept            = default;
ToolCallOutputDecoder& ToolCallOutputDecoder::operator=(ToolCallOutputDecoder&&) noexcept = default;

std::vector<ToolCallDelta> ToolCallOutputDecoder::take_call_deltas() {
    return stream_ ? stream_->take_deltas() : std::vector<ToolCallDelta>{};
}

void ToolCallOutputDecoder::append_region(GeneratedText content, std::size_t from) {
    const std::size_t base = tool_region_.size();
    for (const std::size_t token : content.tool_call_tokens) {
        if (token >= from) { region_tool_calls_.push_back(base + token - from); }
    }
    tool_region_.append(content.text.substr(from));
}

std::string ToolCallOutputDecoder::feed(GeneratedText content) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    const std::string_view text = content.text;
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    fed_content_ = true;
    if (saw_tool_marker_) {
        append_region(content, 0);
        if (stream_) { stream_->feed(text); }
        return {};
    }

    // Constrained output opens calls only with the token its grammar emits; free output also
    // reads the bare agent-harness openers by their spelling.
    const bool bare_markers = !contract_->constrained;
    std::string visible;
    // Every published byte also feeds the code tracker; the held whitespace joins it when it is
    // published, so the tracker always describes the text before the held bytes.
    const auto publish = [&](std::string_view bytes) {
        visible.append(bytes);
        code_.feed(bytes);
        published_tail_.append(bytes);
        if (published_tail_.size() > kToolCallOpen.size()) {
            published_tail_.erase(0, published_tail_.size() - kToolCallOpen.size());
        }
    };
    const auto publish_held = [&] {
        publish(trailing_whitespace_);
        trailing_whitespace_.clear();
        publish(pending_tag_);
        pending_tag_.clear();
    };
    // A quoted marker inside Markdown code of free output is text.
    const auto quoted = [&] {
        MarkdownCodeTracker before_marker = code_;
        before_marker.feed(trailing_whitespace_);
        return bare_markers && before_marker.in_code();
    };
    const auto open_region = [&] {
        region_code_ = code_;
        tool_region_ = std::move(trailing_whitespace_);
        trailing_whitespace_.clear();
        tool_region_.append(pending_tag_);
        pending_tag_.clear();
        saw_tool_marker_ = true;
    };
    auto token = content.tool_call_tokens.begin();
    for (std::size_t index = 0; index < text.size(); ++index) {
        while (token != content.tool_call_tokens.end() && *token < index) { ++token; }
        if (token != content.tool_call_tokens.end() && *token == index) {
            // The model's `<tool_call>` token: a bare-marker prefix held before it was text.
            if (!pending_tag_.empty()) { publish_held(); }
            if (!quoted()) {
                open_region();
                append_region(content, index);
                // A constrained region opens only at the token its grammar emits.
                if (contract_->constrained) {
                    stream_ = std::make_unique<ConstrainedToolStream>(*contract_);
                    stream_->feed(std::string_view(tool_region_).substr(region_tool_calls_.front()));
                }
                break;
            }
            // Quoted: its bytes continue as ordinary text below.
        }

        const char byte = text[index];
        if (!pending_tag_.empty()) {
            pending_tag_.push_back(byte);
            if (matches_bare_marker(pending_tag_)) {
                // Quoted, or inside a `<tool_call>` spelled with ordinary tokens: text.
                if (quoted() || published_tail_ == kToolCallOpen) {
                    publish_held();
                    continue;
                }
                open_region();
                append_region(content, index + 1);
                break;
            }
            if (is_prefix_of_bare_marker(pending_tag_)) { continue; }
            // Publish the bytes before this one; the byte itself may start the next marker.
            pending_tag_.pop_back();
            publish_held();
        }

        if (byte == '<' && bare_markers) {
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
    std::vector<std::size_t> openers;
    for (std::size_t at = prefix.find(kToolCallOpen); at != std::string_view::npos;
         at = prefix.find(kToolCallOpen, at + 1)) {
        openers.push_back(at);
    }
    (void)feed({prefix, openers});
    if (stream_) {
        // The call the prefix began streams with the generation's first output, prefix included,
        // so its deltas still concatenate to its arguments.
        if (!stream_->calls().empty() || stream_->status() == ConstrainedToolStream::Status::Invalid)
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed or malformed calls");
    } else if (saw_tool_marker_) {
        std::vector<RawToolCall> calls;
        (void)QwenToolRegionParser(tool_region_, region_tool_calls_, 0, max_tool_name_length_,
                                   *contract_, tolerant_)
            .parse(calls);
        if (!calls.empty())
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed calls");
    }
    continuation_withheld_bytes_ =
        trailing_whitespace_.size() + pending_tag_.size() + tool_region_.size();
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish(FinishReason reason,
                                                              GeneratedText open_reasoning) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    // Only a turn the model ended itself recovers calls from its open thinking; an output,
    // context or cancellation cut leaves a call it was still deliberating over unexecuted.
    if (tolerant_ && !contract_->constrained && reason == FinishReason::StopToken &&
        !open_reasoning.text.empty() && !fed_content_) {
        ParsedToolCallOutput recovered =
            parse_open_reasoning_calls(open_reasoning, max_tool_name_length_, *contract_);
        if (recovered.is_tool_call_response) {
            return Terminal{.content     = {},
                            .tool_calls  = std::move(recovered.tool_calls),
                            .diagnostics = recovered.diagnostics};
        }
    }

    if (stream_) {
        const auto status      = stream_->status();
        const bool interrupted = reason == FinishReason::OutputLimit ||
                                 reason == FinishReason::ContextCapacity ||
                                 reason == FinishReason::Cancelled;
        if (status == ConstrainedToolStream::Status::Invalid ||
            (!interrupted && status != ConstrainedToolStream::Status::Complete))
            throw std::logic_error("committed tool grammar and parameter decoder disagree");
        Terminal result;
        result.diagnostics             = stream_->diagnostics();
        result.diagnostics.marker_seen = true;
        result.tool_calls              = std::move(stream_->calls());
        return result;
    }
    if (contract_->constrained && contract_->required && reason == FinishReason::StopToken)
        throw std::logic_error("required tool grammar ended without a call");
    if (contract_->constrained && contract_->required) return {};

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output({tool_region_, region_tool_calls_}, max_tool_name_length_,
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
