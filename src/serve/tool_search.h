#pragma once

// Server-side search over deferred tool declarations (Anthropic's tool_search_tool_regex and
// tool_search_tool_bm25). Each tool contributes its name, description, and the names and
// descriptions of its input schema's top-level properties.
//
// The regex variant follows Python re.search with IGNORECASE over each of those fields: a tool
// matches when any field contains a match. Matching runs as a Thompson-NFA simulation, so its cost
// is linear in the field length for every accepted pattern; constructs that need backtracking
// (backreferences, lookaround, possessive quantifiers, atomic groups, conditionals) are refused as
// invalid input instead of being approximated. Case-insensitive comparison uses Unicode simple case
// mappings.
//
// The BM25 variant ranks tools by Okapi BM25 (k1 = 1.2, b = 0.75) over word tokens: maximal runs of
// letters and digits, split again at lower-to-upper case changes, lowercased, with a final plural
// "s" removed from tokens longer than three characters. Name tokens count twice.

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infernix::serve {

// The search input was refused: a malformed or unsupported pattern, or one over the length limit.
class ToolSearchInputError final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// The search exceeded its work budget (pattern program size times searched text).
class ToolSearchBudgetExceeded final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline constexpr std::size_t kToolSearchDefaultLimit      = 5;
inline constexpr std::size_t kToolSearchMaximumLimit      = 10000;
inline constexpr std::size_t kToolSearchMaximumPattern    = 200; // code points
inline constexpr std::size_t kToolSearchMaximumQuery      = 500; // code points
inline constexpr std::size_t kToolSearchMaximumDeferred   = 10000;

class ToolSearchIndex {
public:
    // Indexes `tools` by position; a result is an index into this span.
    explicit ToolSearchIndex(std::span<const ToolDefinition> tools);

    // Tools with a field that contains a match of `pattern`, at most `limit`: name matches first,
    // then the rest, each group in declaration order.
    [[nodiscard]] std::vector<std::size_t> regex(std::string_view pattern, std::size_t limit) const;

    // Tools with a positive BM25 score for `query`, best first (declaration order on ties), at
    // most `limit`.
    [[nodiscard]] std::vector<std::size_t> bm25(std::string_view query, std::size_t limit) const;

    [[nodiscard]] std::size_t size() const noexcept { return documents_.size(); }

    struct Field {
        // The field's code points, and the same code points under simple lowercase mapping.
        std::vector<std::int32_t> text;
        std::vector<std::int32_t> folded;
    };

private:
    struct Document {
        std::vector<Field> fields; // fields[0] is the tool name
        double length = 0.0;       // BM25 token count, name tokens counted twice
    };

    struct Posting {
        std::uint32_t document = 0;
        std::uint32_t count    = 0;
    };

    std::vector<Document> documents_;
    std::unordered_map<std::string, std::uint32_t> term_ids_;
    std::vector<std::vector<Posting>> postings_; // by term id, documents ascending
    double average_length_ = 0.0;
};

} // namespace infernix::serve
