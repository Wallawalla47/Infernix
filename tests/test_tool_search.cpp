// Server-side tool search over deferred declarations: the Python-re subset the regex variant
// accepts (case-insensitive re.search per field, linear-time matching), the constructs it refuses,
// and BM25 ranking. The expected matches follow Python's re module for the same pattern and text.

#include "serve/tool_search.h"

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ToolDefinition tool(std::string name, std::string description, std::string schema) {
    ToolDefinition definition;
    definition.name              = std::move(name);
    definition.description       = std::move(description);
    definition.input_schema_json = std::move(schema);
    definition.deferred          = true;
    return definition;
}

std::vector<ToolDefinition> catalog() {
    return {
        tool("get_weather", "Get the current weather at a location",
             R"({"type":"object","properties":{"location":{"type":"string","description":"City and country"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}}})"),
        tool("search_files", "Search through files in the workspace",
             R"({"type":"object","properties":{"query":{"type":"string"},"file_types":{"type":"array","items":{"type":"string"},"description":"Extensions such as .cpp"}}})"),
        tool("get_user_data", "Read a user's profile",
             R"({"type":"object","properties":{"user_id":{"type":"string"}}})"),
        tool("slackPostMessage", "Post a message to a Slack channel; supports forecasts in "
                                 "threads",
             R"({"type":"object","properties":{"channel":{"type":"string"},"text":{"type":"string"}}})"),
        tool("database_query", "Run a read-only SQL query against the analytics database",
             R"({"type":"object","properties":{"sql":{"type":"string"}}})"),
    };
}

std::vector<std::size_t> regex(const ToolSearchIndex& index, const std::string& pattern,
                               std::size_t limit = 10) {
    return index.regex(pattern, limit);
}

std::string input_error(const std::function<void()>& action) {
    try {
        action();
    } catch (const ToolSearchInputError& error) { return error.what(); } catch (...) {
        return "wrong exception";
    }
    return {};
}

int test_regex_matching() {
    const std::vector<ToolDefinition> tools = catalog();
    const ToolSearchIndex index(tools);
    int failures = 0;
    failures += check(regex(index, "weather") == std::vector<std::size_t>{0},
                      "a literal did not match the name");
    failures += check(regex(index, "WEATHER") == std::vector<std::size_t>{0},
                      "matching was not case-insensitive");
    failures += check(regex(index, "get_.*_data") == std::vector<std::size_t>{2},
                      "a wildcard name pattern did not match");
    // Name matches come first, then description and argument matches, each in declaration order.
    failures += check(regex(index, "forecast|weather") == std::vector<std::size_t>{0, 3},
                      "alternation over names and descriptions ranked wrongly");
    failures += check(regex(index, "database.*query|query.*database") ==
                          std::vector<std::size_t>{4},
                      "either-order alternation did not match");
    failures += check(regex(index, "city and country") == std::vector<std::size_t>{0},
                      "an argument description was not searched");
    failures += check(regex(index, "^file_types$") == std::vector<std::size_t>{1},
                      "an argument name was not searched as its own field");
    failures += check(regex(index, "^get_") == std::vector<std::size_t>{0, 2},
                      "^ did not anchor at a field start");
    failures += check(regex(index, "data$") == std::vector<std::size_t>{2},
                      "$ did not anchor at a field end");
    // "slackPostMessage": no boundary between "slack" and "Post".
    failures += check(regex(index, R"(^slack\b)").empty() &&
                          regex(index, R"(^slack\B)") == std::vector<std::size_t>{3},
                      "word boundaries do not follow Python's \\b");
    failures += check(regex(index, "[A-Z]{4}POST") == std::vector<std::size_t>{3},
                      "a class with a counted repeat did not fold case");
    failures += check(regex(index, R"(sql\b)") == std::vector<std::size_t>{4},
                      "an escape sequence did not match");
    failures += check(regex(index, R"(\.cpp)") == std::vector<std::size_t>{1},
                      "an escaped dot did not match a literal dot");
    failures += check(regex(index, "(?:user)_(id|data)") == std::vector<std::size_t>{2},
                      "groups did not match");
    failures += check(regex(index, "(?-i:WEATHER)").empty() &&
                          regex(index, "(?-i:weather)") == std::vector<std::size_t>{0},
                      "a scoped flag did not switch case sensitivity");
    failures += check(regex(index, "nothing_matches_this").empty(), "a non-match matched");
    failures += check(regex(index, "", 2) == std::vector<std::size_t>{0, 1},
                      "an empty pattern did not match every tool up to the limit");
    failures += check(regex(index, "e", 3).size() == 3, "the limit was not applied");
    return failures;
}

int test_regex_refusals() {
    const std::vector<ToolDefinition> tools = catalog();
    const ToolSearchIndex index(tools);
    int failures = 0;
    for (const std::string pattern :
         {"(?=weather)", "(?!x)y", "(?<=get_)weather", "(a)\\1", "(?P<n>a)(?P=n)", "a*+",
          "(?>a)", "(?(1)a|b)"}) {
        failures += check(input_error([&] { (void)regex(index, pattern); })
                                  .find("Unsupported regular expression pattern") == 0,
                          "a backtracking-only construct was accepted: " + pattern);
    }
    for (const std::string pattern : {"(", "weather)", "[a-", "*a", "a**", "[z-a]", R"(\q)",
                                      "a{3,1}", "a(?i)b"}) {
        failures += check(input_error([&] { (void)regex(index, pattern); })
                                  .find("Invalid regular expression pattern") == 0,
                          "a malformed pattern was accepted: " + pattern);
    }
    failures += check(input_error([&] { (void)regex(index, "missing) at"); }) ==
                          "Invalid regular expression pattern: unbalanced parenthesis at position 7",
                      "the error position is not Python's");
    failures += check(!input_error([&] { (void)regex(index, std::string(201, 'a')); }).empty() &&
                          input_error([&] { (void)regex(index, std::string(200, 'a')); }).empty(),
                      "the 200-character pattern limit is wrong");
    failures += check(!input_error([&] { (void)regex(index, "(?:(?:a{100}){100}){100}"); }).empty(),
                      "an expansion past the state limit compiled");
    failures += check(!input_error([&] { (void)regex(index, "(?:(?:a{0}){10000}){10000}"); }).empty(),
                      "an empty repetition storm compiled without a bound");
    return failures;
}

int test_bm25() {
    const std::vector<ToolDefinition> tools = catalog();
    const ToolSearchIndex index(tools);
    int failures = 0;
    const std::vector<std::size_t> weather = index.bm25("what is the weather forecast", 5);
    failures += check(!weather.empty() && weather.front() == 0,
                      "BM25 did not rank the weather tool first");
    failures += check(index.bm25("post slack messages", 5).front() == 3,
                      "BM25 did not split camelCase names or fold plurals");
    failures += check(index.bm25("SQL database", 1) == std::vector<std::size_t>{4},
                      "BM25 did not rank by term weight or apply the limit");
    failures += check(index.bm25("xylophone", 5).empty(), "BM25 returned unrelated tools");
    failures += check(!input_error([&] { (void)index.bm25("!!!", 5); }).empty(),
                      "a query without words was accepted");
    failures += check(!input_error([&] { (void)index.bm25(std::string(501, 'a'), 5); }).empty(),
                      "the 500-character query limit was not applied");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_regex_matching();
    failures += test_regex_refusals();
    failures += test_bm25();
    if (failures != 0) {
        std::cerr << failures << " tool search check(s) failed\n";
        return 1;
    }
    std::cout << "tool search tests passed\n";
    return 0;
}
