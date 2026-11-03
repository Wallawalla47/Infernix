#pragma once

#include <nlohmann/json.hpp>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::text {
struct ParsedJsonNumbers {
    nlohmann::ordered_json value;
    std::vector<std::string> inexact_numbers;
};

// The deepest nesting of arrays and objects a parsed document may have: the DOM is built
// iteratively, but serializing, copying and walking it recurse once per level.
constexpr std::size_t kMaxJsonNesting = 256;

// A document nested deeper than kMaxJsonNesting.
class JsonNestingError : public std::invalid_argument {
public:
    JsonNestingError()
        : std::invalid_argument("JSON nests arrays or objects deeper than " +
                                std::to_string(kMaxJsonNesting) + " levels") {}
};

// Build the ordinary ordered DOM in one pass, retaining locations whose numeric spelling
// would change value. Consumers apply the precision contract only to active schemas. Throws
// JsonNestingError past kMaxJsonNesting levels.
ParsedJsonNumbers parse_json_numbers(std::string_view source);
std::optional<std::string> inexact_schema_number(const ParsedJsonNumbers& parsed,
                                                 const std::string& schema_pointer = {});
} // namespace infernix::text
