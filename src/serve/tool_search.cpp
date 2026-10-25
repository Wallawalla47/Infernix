#include "serve/tool_search.h"

#include "serve/request_json.h"
#include "text/unicode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

namespace unicode = infernix::text::unicode_internal;

using Field = ToolSearchIndex::Field;

// Instructions a compiled pattern may hold, and simulation steps one search may take. A step is
// one live NFA state at one input position, so the budget bounds the search's time directly.
constexpr std::size_t kMaximumProgram     = 10000;
constexpr std::size_t kMaximumCompileWork = 1'000'000;
constexpr std::uint64_t kMaximumSteps     = 200'000'000;
constexpr std::int32_t kInfiniteRepeat    = -1;
constexpr std::int32_t kMaximumCodePoint  = 0x10ffff;

[[noreturn]] void invalid_pattern(const std::string& message, std::size_t position) {
    throw ToolSearchInputError("Invalid regular expression pattern: " + message + " at position " +
                               std::to_string(position));
}

[[noreturn]] void unsupported_pattern(const std::string& construct, std::size_t position) {
    throw ToolSearchInputError("Unsupported regular expression pattern: " + construct +
                               " at position " + std::to_string(position) +
                               " needs backtracking, which tool search does not run");
}

std::vector<std::int32_t> code_points(std::string_view text) {
    std::vector<std::int32_t> result;
    result.reserve(text.size());
    for (const unicode::CodepointSpan span : unicode::utf8_codepoints(text, "tool search text")) {
        result.push_back(span.value);
    }
    return result;
}

Field make_field(std::string_view text) {
    Field field;
    field.text = code_points(text);
    field.folded.reserve(field.text.size());
    for (const std::int32_t value : field.text) {
        field.folded.push_back(unicode::simple_lowercase(value));
    }
    return field;
}

// Python's str.isspace() set: Unicode Zs/Zl/Zp, the ASCII controls \t..\r, and \x1c-\x1f, \x85.
bool python_space(std::int32_t value) {
    return unicode::is_whitespace(value) || (value >= 0x1c && value <= 0x1f) || value == 0x85;
}

bool python_word(std::int32_t value, bool ascii) {
    if (ascii || value < 0x80) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '_';
    }
    return unicode::is_letter(value) || unicode::is_number(value);
}

bool python_digit(std::int32_t value, bool ascii) {
    if (ascii || value < 0x80) { return value >= '0' && value <= '9'; }
    return unicode::is_decimal_digit(value);
}

bool python_class_space(std::int32_t value, bool ascii) {
    if (ascii || value < 0x80) {
        return value == ' ' || (value >= '\t' && value <= '\r') || (!ascii && value >= 0x1c && value <= 0x1f);
    }
    return python_space(value);
}

// ---------------------------------------------------------------------------------------------
// Pattern representation

enum class Category : std::uint8_t {
    Digit,
    NotDigit,
    Word,
    NotWord,
    Space,
    NotSpace,
};

struct CharClass {
    std::vector<std::pair<std::int32_t, std::int32_t>> ranges;
    std::vector<Category> categories;
    bool negated = false;
    bool fold    = false;
    bool ascii   = false;

    [[nodiscard]] bool contains_exact(std::int32_t value) const {
        for (const auto& [low, high] : ranges) {
            if (value >= low && value <= high) { return true; }
        }
        for (const Category category : categories) {
            switch (category) {
            case Category::Digit:
                if (python_digit(value, ascii)) { return true; }
                break;
            case Category::NotDigit:
                if (!python_digit(value, ascii)) { return true; }
                break;
            case Category::Word:
                if (python_word(value, ascii)) { return true; }
                break;
            case Category::NotWord:
                if (!python_word(value, ascii)) { return true; }
                break;
            case Category::Space:
                if (python_class_space(value, ascii)) { return true; }
                break;
            case Category::NotSpace:
                if (!python_class_space(value, ascii)) { return true; }
                break;
            }
        }
        return false;
    }

    [[nodiscard]] bool matches(std::int32_t value) const {
        bool inside = contains_exact(value);
        if (!inside && fold) {
            const std::int32_t lower = unicode::simple_lowercase(value);
            const std::int32_t upper = unicode::simple_uppercase(value);
            inside = (lower != value && contains_exact(lower)) ||
                     (upper != value && contains_exact(upper));
        }
        return inside != negated;
    }
};

enum class Assertion : std::uint8_t {
    Start,          // \A, and ^ without MULTILINE
    LineStart,      // ^ with MULTILINE
    End,            // \Z
    EndOrFinalLine, // $ without MULTILINE: the end, or before a final newline
    LineEnd,        // $ with MULTILINE
    WordBoundary,
    NotWordBoundary,
};

enum class NodeKind : std::uint8_t {
    Empty,
    Literal,
    Any,
    Class,
    Assert,
    Concat,
    Alternate,
    Repeat,
};

struct Node {
    NodeKind kind = NodeKind::Empty;
    std::int32_t value = 0; // Literal code point
    bool fold          = false;
    bool dotall        = false;
    bool ascii         = false; // word boundaries under the ASCII flag
    std::size_t klass  = 0;     // Class index
    Assertion assertion = Assertion::Start;
    std::vector<std::size_t> children;
    std::int32_t minimum = 0;
    std::int32_t maximum = 0;
};

struct Flags {
    bool ignore_case = true; // the API searches case-insensitively
    bool dotall      = false;
    bool multiline   = false;
    bool verbose     = false;
    bool ascii       = false;
};

class PatternParser {
public:
    explicit PatternParser(std::vector<std::int32_t> pattern) : pattern_(std::move(pattern)) {}

    std::size_t parse() {
        parse_global_flags();
        const std::size_t root = parse_alternation(0);
        if (position_ < pattern_.size()) {
            // Only an unmatched ')' stops the top-level alternation early.
            invalid_pattern("unbalanced parenthesis", position_);
        }
        return root;
    }

    std::vector<Node> nodes;
    std::vector<CharClass> classes;

private:
    std::size_t add(Node node) {
        nodes.push_back(std::move(node));
        return nodes.size() - 1U;
    }

    [[nodiscard]] bool at_end() const { return position_ >= pattern_.size(); }
    [[nodiscard]] std::int32_t peek(std::size_t ahead = 0) const {
        return position_ + ahead < pattern_.size() ? pattern_[position_ + ahead] : -1;
    }

    void skip_verbose() {
        if (!flags_.verbose) { return; }
        while (!at_end()) {
            const std::int32_t value = peek();
            if (python_space(value)) {
                ++position_;
            } else if (value == '#') {
                while (!at_end() && peek() != '\n') { ++position_; }
            } else {
                return;
            }
        }
    }

    static bool flag_letter(std::int32_t value) {
        return value == 'a' || value == 'i' || value == 'L' || value == 'm' || value == 's' ||
               value == 'u' || value == 'x';
    }

    void apply_flag(Flags& flags, std::int32_t value, bool enable, std::size_t at) {
        switch (value) {
        case 'i':
            flags.ignore_case = enable;
            break;
        case 's':
            flags.dotall = enable;
            break;
        case 'm':
            flags.multiline = enable;
            break;
        case 'x':
            flags.verbose = enable;
            break;
        case 'a':
            if (!enable) { invalid_pattern("bad inline flag: cannot turn off flags 'a'", at); }
            flags.ascii = true;
            break;
        case 'u':
            if (!enable) { invalid_pattern("bad inline flag: cannot turn off flags 'u'", at); }
            flags.ascii = false;
            break;
        case 'L':
            invalid_pattern("bad inline flag: cannot use 'locale' flag with a str pattern", at);
        default:
            invalid_pattern("unknown flag", at);
        }
    }

    // Python accepts global flags `(?aiLmsux)` only at the start of the expression.
    void parse_global_flags() {
        while (peek() == '(' && peek(1) == '?' && flag_letter(peek(2))) {
            std::size_t scan = position_ + 2U;
            while (scan < pattern_.size() && flag_letter(pattern_[scan])) { ++scan; }
            if (scan >= pattern_.size() || pattern_[scan] != ')') { return; }
            for (std::size_t index = position_ + 2U; index < scan; ++index) {
                apply_flag(flags_, pattern_[index], true, index);
            }
            position_ = scan + 1U;
            skip_verbose();
        }
    }

    std::size_t parse_alternation(std::size_t depth) {
        if (depth > 200) { invalid_pattern("too many nested groups", position_); }
        std::vector<std::size_t> branches{parse_concatenation(depth)};
        while (peek() == '|') {
            ++position_;
            branches.push_back(parse_concatenation(depth));
        }
        if (branches.size() == 1U) { return branches.front(); }
        Node node;
        node.kind     = NodeKind::Alternate;
        node.children = std::move(branches);
        return add(std::move(node));
    }

    std::size_t parse_concatenation(std::size_t depth) {
        std::vector<std::size_t> items;
        for (;;) {
            skip_verbose();
            if (at_end() || peek() == '|' || peek() == ')') { break; }
            items.push_back(parse_repeat(depth));
        }
        if (items.size() == 1U) { return items.front(); }
        Node node;
        node.kind     = items.empty() ? NodeKind::Empty : NodeKind::Concat;
        node.children = std::move(items);
        return add(std::move(node));
    }

    std::optional<std::int32_t> parse_count() {
        std::optional<std::int32_t> count;
        while (!at_end() && peek() >= '0' && peek() <= '9') {
            const std::int64_t next =
                static_cast<std::int64_t>(count.value_or(0)) * 10 + (peek() - '0');
            if (next > std::numeric_limits<std::int32_t>::max()) {
                invalid_pattern("the repetition number is too large", position_);
            }
            count = static_cast<std::int32_t>(next);
            ++position_;
        }
        return count;
    }

    // `{m}`, `{m,}`, `{,n}`, `{m,n}`; anything else is a literal '{', as in Python.
    bool parse_braces(std::int32_t& minimum, std::int32_t& maximum) {
        const std::size_t start = position_;
        ++position_;
        const std::optional<std::int32_t> low = parse_count();
        std::optional<std::int32_t> high      = low;
        bool comma                             = false;
        if (peek() == ',') {
            comma = true;
            ++position_;
            high = parse_count();
        }
        if (peek() != '}' || (!low && !comma)) {
            position_ = start;
            return false;
        }
        ++position_;
        minimum = low.value_or(0);
        maximum = comma && !high ? kInfiniteRepeat : *high;
        if (maximum != kInfiniteRepeat && maximum < minimum) {
            invalid_pattern("min repeat greater than max repeat", start + 1U);
        }
        return true;
    }

    std::size_t parse_repeat(std::size_t depth) {
        const std::size_t atom_start = position_;
        std::size_t atom             = parse_atom(depth);
        bool repeated                = false;
        for (;;) {
            skip_verbose();
            const std::size_t at = position_;
            std::int32_t minimum = 0;
            std::int32_t maximum = 0;
            const std::int32_t value = peek();
            if (value == '*') {
                minimum = 0;
                maximum = kInfiniteRepeat;
                ++position_;
            } else if (value == '+') {
                minimum = 1;
                maximum = kInfiniteRepeat;
                ++position_;
            } else if (value == '?') {
                minimum = 0;
                maximum = 1;
                ++position_;
            } else if (value == '{') {
                if (!parse_braces(minimum, maximum)) { break; }
            } else {
                break;
            }
            if (repeated) { invalid_pattern("multiple repeat", at); }
            const NodeKind kind = nodes[atom].kind;
            if (kind == NodeKind::Empty || kind == NodeKind::Assert) {
                invalid_pattern("nothing to repeat", atom_start);
            }
            if (peek() == '+') { unsupported_pattern("possessive repeat", position_); }
            if (peek() == '?') { ++position_; } // lazy: the same matches exist
            Node node;
            node.kind     = NodeKind::Repeat;
            node.children = {atom};
            node.minimum  = minimum;
            node.maximum  = maximum;
            atom          = add(std::move(node));
            repeated      = true;
        }
        return atom;
    }

    std::size_t literal(std::int32_t value) {
        Node node;
        node.kind  = NodeKind::Literal;
        node.value = flags_.ignore_case ? unicode::simple_lowercase(value) : value;
        node.fold  = flags_.ignore_case;
        return add(std::move(node));
    }

    std::size_t assertion(Assertion kind) {
        Node node;
        node.kind      = NodeKind::Assert;
        node.assertion = kind;
        node.ascii     = flags_.ascii;
        return add(std::move(node));
    }

    std::size_t category_class(Category category) {
        CharClass klass;
        klass.categories.push_back(category);
        klass.ascii = flags_.ascii;
        return class_node(std::move(klass));
    }

    std::size_t class_node(CharClass klass) {
        classes.push_back(std::move(klass));
        Node node;
        node.kind  = NodeKind::Class;
        node.klass = classes.size() - 1U;
        return add(std::move(node));
    }

    std::size_t parse_atom(std::size_t depth) {
        const std::size_t start  = position_;
        const std::int32_t value = peek();
        switch (value) {
        case '*':
        case '+':
        case '?':
            invalid_pattern("nothing to repeat", start);
        case '(':
            return parse_group(depth);
        case '[':
            return parse_class();
        case '.': {
            ++position_;
            Node node;
            node.kind   = NodeKind::Any;
            node.dotall = flags_.dotall;
            return add(std::move(node));
        }
        case '^':
            ++position_;
            return assertion(flags_.multiline ? Assertion::LineStart : Assertion::Start);
        case '$':
            ++position_;
            return assertion(flags_.multiline ? Assertion::LineEnd : Assertion::EndOrFinalLine);
        case '\\':
            return parse_escape();
        default:
            ++position_;
            return literal(value);
        }
    }

    std::size_t parse_group(std::size_t depth) {
        const std::size_t start = position_;
        ++position_; // '('
        Flags saved = flags_;
        if (peek() == '?') {
            ++position_;
            const std::int32_t kind = peek();
            if (kind == ':') {
                ++position_;
            } else if (kind == 'P' && peek(1) == '<') {
                position_ += 2U;
                parse_group_name(start);
            } else if (kind == 'P' && peek(1) == '=') {
                unsupported_pattern("backreference", start);
            } else if (kind == '<' && (peek(1) == '=' || peek(1) == '!')) {
                unsupported_pattern("lookbehind", start);
            } else if (kind == '<') {
                ++position_;
                parse_group_name(start);
            } else if (kind == '=' || kind == '!') {
                unsupported_pattern("lookahead", start);
            } else if (kind == '>') {
                unsupported_pattern("atomic group", start);
            } else if (kind == '(') {
                unsupported_pattern("conditional group", start);
            } else if (kind == '#') {
                while (!at_end() && peek() != ')') { ++position_; }
                if (at_end()) { invalid_pattern("missing ), unterminated comment", start); }
                ++position_;
                Node node;
                return add(std::move(node));
            } else if (flag_letter(kind) || kind == '-') {
                parse_scoped_flags(start);
            } else {
                invalid_pattern("unknown extension ?" + std::string(1, static_cast<char>(kind)),
                                start + 1U);
            }
        }
        const std::size_t inner = parse_alternation(depth + 1U);
        if (peek() != ')') { invalid_pattern("missing ), unterminated subpattern", start); }
        ++position_;
        flags_ = saved;
        return inner;
    }

    void parse_group_name(std::size_t start) {
        const std::size_t name_start = position_;
        while (!at_end() && peek() != '>') { ++position_; }
        if (at_end()) { invalid_pattern("missing >, unterminated name", name_start); }
        if (position_ == name_start) { invalid_pattern("missing group name", name_start); }
        for (std::size_t index = name_start; index < position_; ++index) {
            if (!python_word(pattern_[index], false)) {
                invalid_pattern("bad character in group name", name_start);
            }
        }
        (void)start;
        ++position_; // '>'
    }

    // `(?flags-flags:...)`. A flag group without ':' after the start is a misplaced global flag.
    void parse_scoped_flags(std::size_t start) {
        bool enable = true;
        while (!at_end() && peek() != ':' && peek() != ')') {
            const std::int32_t value = peek();
            if (value == '-') {
                if (!enable) { invalid_pattern("bad inline flag", position_); }
                enable = false;
            } else if (flag_letter(value)) {
                if (!enable && (value == 'a' || value == 'u' || value == 'L')) {
                    invalid_pattern("bad inline flag: cannot turn off flags 'a', 'u' and 'L'",
                                    position_);
                }
                apply_flag(flags_, value, enable, position_);
            } else {
                invalid_pattern("unknown flag", position_);
            }
            ++position_;
        }
        if (peek() != ':') { invalid_pattern("global flags not at the start of the expression", start); }
        ++position_;
    }

    std::int32_t parse_hex(std::size_t digits, std::size_t start, char letter) {
        std::int32_t value = 0;
        for (std::size_t index = 0; index < digits; ++index) {
            const std::int32_t digit = peek();
            int nibble               = -1;
            if (digit >= '0' && digit <= '9') {
                nibble = digit - '0';
            } else if (digit >= 'a' && digit <= 'f') {
                nibble = digit - 'a' + 10;
            } else if (digit >= 'A' && digit <= 'F') {
                nibble = digit - 'A' + 10;
            }
            if (nibble < 0) {
                invalid_pattern(std::string("incomplete escape \\") + letter, start);
            }
            value = value * 16 + nibble;
            ++position_;
        }
        if (value > kMaximumCodePoint) {
            invalid_pattern(std::string("bad escape \\") + letter, start);
        }
        return value;
    }

    // An escape that denotes one code point, shared by atoms and classes; nullopt otherwise.
    std::optional<std::int32_t> character_escape(std::int32_t value, std::size_t start,
                                                 bool in_class) {
        switch (value) {
        case 'n':
            return '\n';
        case 't':
            return '\t';
        case 'r':
            return '\r';
        case 'f':
            return '\f';
        case 'v':
            return '\v';
        case 'a':
            return 0x07;
        case 'b':
            if (in_class) { return 0x08; }
            return std::nullopt;
        case 'x':
            return parse_hex(2, start, 'x');
        case 'u':
            return parse_hex(4, start, 'u');
        case 'U':
            return parse_hex(8, start, 'U');
        case 'N':
            unsupported_pattern("named character escape \\N", start);
        default:
            break;
        }
        if (value == '0' || (in_class && value >= '1' && value <= '7')) {
            // Octal: up to three digits.
            std::int32_t result = value - '0';
            for (int count = 1; count < 3 && peek() >= '0' && peek() <= '7'; ++count) {
                result = result * 8 + (peek() - '0');
                ++position_;
            }
            if (result > 0377) { invalid_pattern("octal escape value outside of range 0-0o377", start); }
            return result;
        }
        if (value >= '1' && value <= '9') { unsupported_pattern("backreference", start); }
        if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z')) {
            invalid_pattern("bad escape \\" + std::string(1, static_cast<char>(value)), start);
        }
        return value; // escaped punctuation or a non-ASCII character stands for itself
    }

    std::optional<Category> category_escape(std::int32_t value) const {
        switch (value) {
        case 'd':
            return Category::Digit;
        case 'D':
            return Category::NotDigit;
        case 'w':
            return Category::Word;
        case 'W':
            return Category::NotWord;
        case 's':
            return Category::Space;
        case 'S':
            return Category::NotSpace;
        default:
            return std::nullopt;
        }
    }

    std::size_t parse_escape() {
        const std::size_t start = position_;
        ++position_;
        if (at_end()) { invalid_pattern("bad escape (end of pattern)", start); }
        const std::int32_t value = peek();
        ++position_;
        if (const std::optional<Category> category = category_escape(value)) {
            return category_class(*category);
        }
        switch (value) {
        case 'A':
            return assertion(Assertion::Start);
        case 'Z':
            return assertion(Assertion::End);
        case 'b':
            return assertion(Assertion::WordBoundary);
        case 'B':
            return assertion(Assertion::NotWordBoundary);
        default:
            break;
        }
        return literal(*character_escape(value, start, false));
    }

    std::size_t parse_class() {
        const std::size_t start = position_;
        ++position_; // '['
        CharClass klass;
        klass.fold  = flags_.ignore_case;
        klass.ascii = flags_.ascii;
        if (peek() == '^') {
            klass.negated = true;
            ++position_;
        }
        bool first = true;
        for (;;) {
            if (at_end()) { invalid_pattern("unterminated character set", start); }
            const std::size_t item_start = position_;
            std::int32_t value           = peek();
            if (value == ']' && !first) {
                ++position_;
                break;
            }
            first = false;
            // One class member: a code point, or a category escape that cannot bound a range.
            std::optional<std::int32_t> low;
            ++position_;
            if (value == '\\') {
                if (at_end()) { invalid_pattern("unterminated character set", start); }
                const std::int32_t escaped = peek();
                ++position_;
                if (const std::optional<Category> category = category_escape(escaped)) {
                    klass.categories.push_back(*category);
                    if (peek() == '-' && peek(1) != ']' && peek(1) != -1) {
                        invalid_pattern("bad character range", item_start);
                    }
                    continue;
                }
                low = character_escape(escaped, item_start, true);
            } else {
                low = value;
            }
            if (peek() == '-' && peek(1) != ']' && peek(1) != -1) {
                ++position_; // '-'
                const std::size_t high_start = position_;
                std::int32_t high            = peek();
                ++position_;
                if (high == '\\') {
                    if (at_end()) { invalid_pattern("unterminated character set", start); }
                    const std::int32_t escaped = peek();
                    ++position_;
                    if (category_escape(escaped)) {
                        invalid_pattern("bad character range", item_start);
                    }
                    high = *character_escape(escaped, high_start, true);
                }
                if (high < *low) { invalid_pattern("bad character range", item_start); }
                klass.ranges.emplace_back(*low, high);
            } else {
                klass.ranges.emplace_back(*low, *low);
            }
        }
        return class_node(std::move(klass));
    }

    std::vector<std::int32_t> pattern_;
    std::size_t position_ = 0;
    Flags flags_;
};

// ---------------------------------------------------------------------------------------------
// Thompson-NFA program and simulation

enum class Op : std::uint8_t {
    Literal,
    FoldedLiteral,
    Any,
    AnyNewline,
    Class,
    Assert,
    Split,
    Jump,
    Match,
};

struct Instruction {
    Op op               = Op::Match;
    std::int32_t value  = 0;
    bool ascii          = false;
    Assertion assertion = Assertion::Start;
    std::uint32_t x     = 0;
    std::uint32_t y     = 0;
};

class Compiler {
public:
    Compiler(const std::vector<Node>& nodes) : nodes_(nodes) {}

    std::vector<Instruction> compile(std::size_t root) {
        emit_node(root);
        emit({.op = Op::Match});
        return std::move(program_);
    }

private:
    std::uint32_t here() const { return static_cast<std::uint32_t>(program_.size()); }

    std::uint32_t emit(Instruction instruction) {
        if (program_.size() >= kMaximumProgram) {
            throw ToolSearchInputError("Invalid regular expression pattern: the pattern expands "
                                       "to more than " +
                                       std::to_string(kMaximumProgram) + " NFA states");
        }
        program_.push_back(instruction);
        return here() - 1U;
    }

    void emit_node(std::size_t index) {
        // Repeats of subpatterns that emit nothing (`(?:a{0}){1000000}`) cost work, not states.
        if (++work_ > kMaximumCompileWork) {
            throw ToolSearchInputError("Invalid regular expression pattern: the pattern's "
                                       "repetitions are too large to compile");
        }
        const Node& node = nodes_[index];
        switch (node.kind) {
        case NodeKind::Empty:
            return;
        case NodeKind::Literal:
            emit({.op = node.fold ? Op::FoldedLiteral : Op::Literal, .value = node.value});
            return;
        case NodeKind::Any:
            emit({.op = node.dotall ? Op::AnyNewline : Op::Any});
            return;
        case NodeKind::Class:
            emit({.op = Op::Class, .value = static_cast<std::int32_t>(node.klass)});
            return;
        case NodeKind::Assert:
            emit({.op = Op::Assert, .ascii = node.ascii, .assertion = node.assertion});
            return;
        case NodeKind::Concat:
            for (const std::size_t child : node.children) { emit_node(child); }
            return;
        case NodeKind::Alternate: {
            std::vector<std::uint32_t> exits;
            for (std::size_t branch = 0; branch < node.children.size(); ++branch) {
                if (branch + 1U == node.children.size()) {
                    emit_node(node.children[branch]);
                    break;
                }
                const std::uint32_t split = emit({.op = Op::Split});
                program_[split].x         = here();
                emit_node(node.children[branch]);
                exits.push_back(emit({.op = Op::Jump}));
                program_[split].y = here();
            }
            for (const std::uint32_t jump : exits) { program_[jump].x = here(); }
            return;
        }
        case NodeKind::Repeat:
            emit_repeat(node);
            return;
        }
    }

    void emit_repeat(const Node& node) {
        const std::size_t child = node.children.front();
        for (std::int32_t count = 0; count < node.minimum; ++count) { emit_node(child); }
        if (node.maximum == kInfiniteRepeat) {
            const std::uint32_t loop = emit({.op = Op::Split});
            program_[loop].x         = here();
            emit_node(child);
            emit({.op = Op::Jump, .x = loop});
            program_[loop].y = here();
            return;
        }
        std::vector<std::uint32_t> skips;
        for (std::int32_t count = node.minimum; count < node.maximum; ++count) {
            const std::uint32_t split = emit({.op = Op::Split});
            program_[split].x         = here();
            skips.push_back(split);
            emit_node(child);
        }
        for (const std::uint32_t split : skips) { program_[split].y = here(); }
    }

    const std::vector<Node>& nodes_;
    std::vector<Instruction> program_;
    std::size_t work_ = 0;
};

class SparseSet {
public:
    explicit SparseSet(std::size_t capacity) : dense_(capacity), sparse_(capacity) {}

    bool insert(std::uint32_t value) {
        if (contains(value)) { return false; }
        sparse_[value]  = static_cast<std::uint32_t>(size_);
        dense_[size_++] = value;
        return true;
    }

    [[nodiscard]] bool contains(std::uint32_t value) const {
        const std::uint32_t slot = sparse_[value];
        return slot < size_ && dense_[slot] == value;
    }

    void clear() noexcept { size_ = 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint32_t operator[](std::size_t index) const { return dense_[index]; }

private:
    std::vector<std::uint32_t> dense_;
    std::vector<std::uint32_t> sparse_;
    std::size_t size_ = 0;
};

class Matcher {
public:
    Matcher(const std::vector<Instruction>& program, const std::vector<CharClass>& classes)
        : program_(program), classes_(classes), current_(program.size()), next_(program.size()) {
        stack_.reserve(program.size());
    }

    // Whether `field` contains a match anywhere (re.search).
    bool search(const Field& field, std::uint64_t& steps) {
        const std::size_t length = field.text.size();
        current_.clear();
        for (std::size_t position = 0;; ++position) {
            if (add(current_, 0, field, position)) { return true; }
            if (position == length) { return false; }
            steps += current_.size();
            if (steps > kMaximumSteps) {
                throw ToolSearchBudgetExceeded("tool search exceeded its execution budget");
            }
            next_.clear();
            const std::int32_t text   = field.text[position];
            const std::int32_t folded = field.folded[position];
            for (std::size_t index = 0; index < current_.size(); ++index) {
                const std::uint32_t pc         = current_[index];
                const Instruction& instruction = program_[pc];
                bool advance                   = false;
                switch (instruction.op) {
                case Op::Literal:
                    advance = text == instruction.value;
                    break;
                case Op::FoldedLiteral:
                    advance = folded == instruction.value;
                    break;
                case Op::Any:
                    advance = text != '\n';
                    break;
                case Op::AnyNewline:
                    advance = true;
                    break;
                case Op::Class:
                    advance = classes_[static_cast<std::size_t>(instruction.value)].matches(text);
                    break;
                default:
                    break;
                }
                if (advance && add(next_, pc + 1U, field, position + 1U)) { return true; }
            }
            std::swap(current_, next_);
        }
    }

private:
    static bool holds(const Instruction& instruction, const Field& field, std::size_t position) {
        const std::size_t length = field.text.size();
        const auto word          = [&](std::size_t at) {
            return python_word(field.text[at], instruction.ascii);
        };
        switch (instruction.assertion) {
        case Assertion::Start:
            return position == 0;
        case Assertion::LineStart:
            return position == 0 || field.text[position - 1U] == '\n';
        case Assertion::End:
            return position == length;
        case Assertion::EndOrFinalLine:
            return position == length ||
                   (position + 1U == length && field.text[position] == '\n');
        case Assertion::LineEnd:
            return position == length || field.text[position] == '\n';
        case Assertion::WordBoundary:
        case Assertion::NotWordBoundary: {
            const bool before   = position > 0 && word(position - 1U);
            const bool after    = position < length && word(position);
            const bool boundary = before != after;
            return instruction.assertion == Assertion::WordBoundary ? boundary : !boundary;
        }
        }
        return false;
    }

    // Follows the empty transitions from `start`; true when a Match state is reached.
    bool add(SparseSet& set, std::uint32_t start, const Field& field, std::size_t position) {
        stack_.clear();
        stack_.push_back(start);
        while (!stack_.empty()) {
            const std::uint32_t pc = stack_.back();
            stack_.pop_back();
            if (!set.insert(pc)) { continue; }
            const Instruction& instruction = program_[pc];
            switch (instruction.op) {
            case Op::Match:
                return true;
            case Op::Jump:
                stack_.push_back(instruction.x);
                break;
            case Op::Split:
                stack_.push_back(instruction.y);
                stack_.push_back(instruction.x);
                break;
            case Op::Assert:
                if (holds(instruction, field, position)) { stack_.push_back(pc + 1U); }
                break;
            default:
                break;
            }
        }
        return false;
    }

    const std::vector<Instruction>& program_;
    const std::vector<CharClass>& classes_;
    SparseSet current_;
    SparseSet next_;
    std::vector<std::uint32_t> stack_;
};

// ---------------------------------------------------------------------------------------------
// BM25 tokens

bool lowercase_letter(std::int32_t value) {
    return unicode::simple_uppercase(value) != value && unicode::simple_lowercase(value) == value;
}

bool uppercase_letter(std::int32_t value) {
    return unicode::simple_lowercase(value) != value && unicode::simple_uppercase(value) == value;
}

std::string normalize_token(const std::vector<std::int32_t>& token) {
    std::size_t length = token.size();
    // A final plural "s" (not "ss") is removed from tokens longer than three characters.
    if (length > 3U && token[length - 1U] == 's' && token[length - 2U] != 's') { --length; }
    std::string result;
    for (std::size_t index = 0; index < length; ++index) {
        result += unicode::codepoint_to_utf8(token[index]);
    }
    return result;
}

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> token;
    std::int32_t previous = 0;
    const auto flush      = [&] {
        if (!token.empty()) { tokens.push_back(normalize_token(token)); }
        token.clear();
    };
    for (const unicode::CodepointSpan span : unicode::utf8_codepoints(text, "tool search text")) {
        const std::int32_t value = span.value;
        const bool part =
            unicode::is_letter(value) || unicode::is_number(value) || unicode::is_mark(value);
        if (!part) {
            flush();
            previous = 0;
            continue;
        }
        if (!token.empty() && lowercase_letter(previous) && uppercase_letter(value)) { flush(); }
        token.push_back(unicode::simple_lowercase(value));
        previous = value;
    }
    flush();
    return tokens;
}

// The searched fields of one declaration: its name, description, and the names and descriptions
// of its input schema's top-level properties.
std::vector<std::string> searched_fields(const ToolDefinition& tool) {
    std::vector<std::string> fields{tool.name};
    if (!tool.description.empty()) { fields.push_back(tool.description); }
    const RequestJson schema = RequestJson::parse(tool.input_schema_json, nullptr, false);
    if (!schema.is_object() || !schema.contains("properties") ||
        !schema.at("properties").is_object()) {
        return fields;
    }
    for (const auto& [name, property] : schema.at("properties").items()) {
        fields.push_back(name);
        if (property.is_object() && property.contains("description") &&
            property.at("description").is_string()) {
            fields.push_back(property.at("description").get<std::string>());
        }
    }
    return fields;
}

} // namespace

ToolSearchIndex::ToolSearchIndex(std::span<const ToolDefinition> tools) {
    documents_.reserve(tools.size());
    double total_length = 0.0;
    for (std::size_t index = 0; index < tools.size(); ++index) {
        Document document;
        std::unordered_map<std::uint32_t, std::uint32_t> counts;
        const std::vector<std::string> fields = searched_fields(tools[index]);
        for (std::size_t field = 0; field < fields.size(); ++field) {
            document.fields.push_back(make_field(fields[field]));
            const std::uint32_t weight = field == 0 ? 2U : 1U;
            for (const std::string& token : tokenize(fields[field])) {
                const auto [position, inserted] =
                    term_ids_.emplace(token, static_cast<std::uint32_t>(term_ids_.size()));
                if (inserted) { postings_.emplace_back(); }
                counts[position->second] += weight;
                document.length += weight;
            }
        }
        for (const auto& [term, count] : counts) {
            postings_[term].push_back(
                Posting{.document = static_cast<std::uint32_t>(index), .count = count});
        }
        total_length += document.length;
        documents_.push_back(std::move(document));
    }
    average_length_ = documents_.empty() ? 0.0 : total_length / static_cast<double>(documents_.size());
}

std::vector<std::size_t> ToolSearchIndex::regex(std::string_view pattern, std::size_t limit) const {
    std::vector<std::int32_t> points;
    try {
        points = code_points(pattern);
    } catch (const std::invalid_argument& error) { throw ToolSearchInputError(error.what()); }
    if (points.size() > kToolSearchMaximumPattern) {
        throw ToolSearchInputError("Invalid regular expression pattern: the pattern has " +
                                   std::to_string(points.size()) + " characters, more than " +
                                   std::to_string(kToolSearchMaximumPattern));
    }
    PatternParser parser(std::move(points));
    const std::size_t root                 = parser.parse();
    const std::vector<Instruction> program = Compiler(parser.nodes).compile(root);
    Matcher matcher(program, parser.classes);

    std::vector<std::size_t> by_name;
    std::vector<std::size_t> by_other;
    std::uint64_t steps = 0;
    for (std::size_t index = 0; index < documents_.size(); ++index) {
        const std::vector<Field>& fields = documents_[index].fields;
        if (matcher.search(fields.front(), steps)) {
            by_name.push_back(index);
            continue;
        }
        for (std::size_t field = 1; field < fields.size(); ++field) {
            if (matcher.search(fields[field], steps)) {
                by_other.push_back(index);
                break;
            }
        }
    }
    by_name.insert(by_name.end(), by_other.begin(), by_other.end());
    if (by_name.size() > limit) { by_name.resize(limit); }
    return by_name;
}

std::vector<std::size_t> ToolSearchIndex::bm25(std::string_view query, std::size_t limit) const {
    std::vector<std::string> terms;
    try {
        if (code_points(query).size() > kToolSearchMaximumQuery) {
            throw ToolSearchInputError("the query has more than " +
                                       std::to_string(kToolSearchMaximumQuery) + " characters");
        }
        terms = tokenize(query);
    } catch (const ToolSearchInputError&) { throw; } catch (const std::invalid_argument& error) {
        throw ToolSearchInputError(error.what());
    }
    if (terms.empty()) { throw ToolSearchInputError("the query contains no searchable words"); }
    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

    constexpr double k1 = 1.2;
    constexpr double b  = 0.75;
    const double count  = static_cast<double>(documents_.size());
    std::vector<double> scores(documents_.size(), 0.0);
    for (const std::string& term : terms) {
        const auto found = term_ids_.find(term);
        if (found == term_ids_.end()) { continue; }
        const std::vector<Posting>& postings = postings_[found->second];
        const double frequency               = static_cast<double>(postings.size());
        const double idf = std::log(1.0 + (count - frequency + 0.5) / (frequency + 0.5));
        for (const Posting& posting : postings) {
            const double tf     = static_cast<double>(posting.count);
            const double length = documents_[posting.document].length;
            const double norm =
                average_length_ > 0.0 ? 1.0 - b + b * length / average_length_ : 1.0;
            scores[posting.document] += idf * tf * (k1 + 1.0) / (tf + k1 * norm);
        }
    }
    std::vector<std::size_t> ranked;
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (scores[index] > 0.0) { ranked.push_back(index); }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](std::size_t left, std::size_t right) { return scores[left] > scores[right]; });
    if (ranked.size() > limit) { ranked.resize(limit); }
    return ranked;
}

} // namespace infernix::serve
