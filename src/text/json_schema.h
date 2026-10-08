#pragma once

#include <string>
#include <string_view>
#include "infernix/types.h"

namespace xgrammar {
class Grammar;
}

namespace infernix::text {

// Validate the supported JSON Schema dialect, retaining property order and literal data.
// Returns normalized schema text for the vendor converter; errors carry a JSON Pointer.
std::string prepare_json_schema(std::string_view source);
xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint);

} // namespace infernix::text
