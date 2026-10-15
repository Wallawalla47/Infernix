#ifndef XGRAMMAR_JSON_STRING_GRAMMAR_H_
#define XGRAMMAR_JSON_STRING_GRAMMAR_H_

#include <xgrammar/grammar.h>
#include "fsm.h"
#include "grammar_builder.h"

#include <string>
#include <vector>

namespace xgrammar {
int32_t AddScalarStringFSM(GrammarBuilder& builder, const FSMWithStartEnd& fsm,
                           const std::string& rule_name, bool close_json_string,
                           bool share_continuations);
Grammar StringConstraints(const std::vector<std::string>& patterns, int minimum, int maximum,
                          const std::vector<std::string>& excluded, bool json_encoding);

// JSON-encoded string body matching a schema pattern; the caller supplies quotes.
Grammar JSONStringPattern(const std::string& pattern);
// Search semantics and ECMAScript character classes, before JSON string encoding.
std::string SchemaStringPattern(const std::string& pattern);
// Complete JSON string, including quotes, bounded by decoded Unicode scalar count.
Grammar JSONStringLength(int minimum, int maximum);
Grammar JSONStringExcept(const std::vector<std::string>& excluded);
}  // namespace xgrammar
#endif
