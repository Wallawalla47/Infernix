#include "text/grammar.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Json       = nlohmann::ordered_json;
using Constraint = infernix::OutputConstraint;

infernix::text::GrammarCompiler make_compiler() {
    std::vector<std::string> vocabulary;
    for (int i = 0; i < 256; ++i) vocabulary.emplace_back(1, static_cast<char>(i));
    vocabulary.emplace_back();
    return {std::move(vocabulary), {256}, 4 * 1024 * 1024};
}

bool accepts(infernix::text::GrammarCompiler& compiler, const Constraint& constraint,
             const std::string& candidate) {
    auto matcher = compiler.compile(constraint, {}, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    const auto allows = [&](int token) {
        return matcher->masks({}, words) == 0 && (words[token / 32] & (1u << (token % 32)));
    };
    for (unsigned char byte : candidate) {
        if (!allows(byte)) return false;
        matcher->accept(byte);
        matcher->confirm();
    }
    return allows(256);
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void length_boundaries(infernix::text::GrammarCompiler& compiler) {
    for (int upper : {0, 1, 127, 128, 129, 200}) {
        for (int lower : {0, upper}) {
            auto schema = Json{{"type", "string"}, {"minLength", lower}, {"maxLength", upper}};
            const auto constraint = Constraint::json_schema(schema.dump());
            for (int count : {std::max(0, upper - 1), upper, upper + 1}) {
                const bool expected = lower <= count && count <= upper;
                check(accepts(compiler, constraint, Json(std::string(count, 'a')).dump()) ==
                          expected,
                      "bounded JSON string accepted the wrong length");
            }
            if (upper == 0) continue;
            for (const auto& last :
                 std::vector<std::string>{"你", "😀", "\n", "\"", "\\", std::string(1, '\0')}) {
                const std::string prefix(upper - 1, 'a');
                check(accepts(compiler, constraint, Json(prefix + last).dump()),
                      "last JSON scalar did not fit its length budget");
                check(!accepts(compiler, constraint, Json(prefix + last + "a").dump()),
                      "JSON escape or UTF-8 exceeded its length budget");
            }
        }
    }
    const auto minimum = Constraint::json_schema(R"({"type":"string","minLength":129})");
    check(!accepts(compiler, minimum, Json(std::string(128, 'a')).dump()) &&
              accepts(compiler, minimum, Json(std::string(200, 'a')).dump()),
          "unbounded JSON length lost its minimum");

    const auto shared = Constraint::json_schema(R"({"type":"object","properties":{
        "a":{"type":"string","minLength":1,"maxLength":2},
        "b":{"type":"string","minLength":3,"maxLength":4}},
        "required":["a","b"],"additionalProperties":false})");
    check(accepts(compiler, shared, R"({"a":"你","b":"a\nb"})") &&
              !accepts(compiler, shared, R"({"a":"abc","b":"abc"})") &&
              !accepts(compiler, shared, R"({"a":"a","b":"ab"})"),
          "shared JSON scalar rules confused field length bounds");
}

void dense_length_masks() {
    std::vector<std::string> vocabulary{"\"", "a\"", "aa\"", "aaaa\"", "\"a"};
    for (char a = 'a'; a <= 'z'; ++a)
        for (char b = 'a'; b <= 'z'; ++b)
            for (char c = 'a'; c <= 'z'; ++c) vocabulary.push_back(std::string{a, b, c});
    const int eos = static_cast<int>(vocabulary.size());
    vocabulary.emplace_back();
    infernix::text::GrammarCompiler compiler(vocabulary, {eos}, 8 * 1024 * 1024);
    const auto constraint =
        Constraint::json_schema(R"({"type":"string","minLength":127,"maxLength":129})");
    for (int consumed : {0, 126, 127, 128, 129}) {
        auto session = compiler.compile(constraint, {}, "\"" + std::string(consumed, 'a'));
        std::vector<std::uint32_t> words(session->mask_words());
        check(session->masks({}, words) == 0, "length mask reached a dead end");
        for (int token = 0; token <= eos; ++token) {
            bool expected = token >= 5 && token < eos && consumed + 3 <= 129;
            if (token < 4) {
                const int length = consumed + static_cast<int>(vocabulary[token].size()) - 1;
                expected         = length >= 127 && length <= 129;
            }
            check(bool(words[token / 32] & (1u << (token % 32))) == expected,
                  "dense JSON mask differs from independent length bounds");
        }
    }
}

void contracts(infernix::text::GrammarCompiler& compiler) {
    auto object = Constraint::json_object();
    for (const std::string text : {"{}", R"({"x":[1,true,null,{"y":"你好"}]})"})
        check(accepts(compiler, object, text), "JSON object rejected valid content");
    for (const std::string text : {"[]", "1", "{}{}", R"({"x":})"})
        check(!accepts(compiler, object, text), "JSON object admitted invalid content");
    // The layout a model writes on its own: with only compact JSON admitted, Qwen3.8 without thinking
    // closed the first key empty (`{"":"ok"}`, NInfer #392). Whitespace runs are bounded at 32.
    const std::string pretty = "{\n  \"status\": \"ok\",\n  \"list\": [\n    1,\n    2\n  ]\n}";
    check(accepts(compiler, object, pretty), "JSON object rejected pretty-printed content");
    check(accepts(compiler, object, "{" + std::string(32, ' ') + "\"x\":1}"),
          "JSON object rejected a 32-character whitespace run");
    check(!accepts(compiler, object, "{" + std::string(33, ' ') + "\"x\":1}"),
          "JSON object admitted an unbounded whitespace run");
    auto schema = Constraint::json_schema(
        R"({"type":"object","properties":{"description":{"const":"first"},"second":{"const":{"description":"second"}}},"required":["description","second"],"additionalProperties":false})");
    check(accepts(compiler, schema, R"({"description":"first","second":{"description":"second"}})"),
          "annotation-named properties were lost");
    check(!accepts(compiler, schema, R"({"description":"first","second":{"description":"first"}})"),
          "literal data was aliased in schema cache");
    auto length = Constraint::json_schema(R"({"type":"string","minLength":1,"maxLength":1})");
    for (const std::string value :
         std::vector<std::string>{"你", "😀", "\n", "\t", std::string(1, '\0'), "\"", "\\"})
        check(accepts(compiler, length, Json(value).dump()),
              "JSON character length/encoding mismatch");
    check(!accepts(compiler, length, Json("ab").dump()), "string length was ignored");
    auto pattern = Constraint::json_schema(R"({"type":"string","pattern":"p"})");
    check(accepts(compiler, pattern, "\"apple\""), "pattern must use search semantics");
    check(!accepts(compiler, pattern, "\"absent\""), "pattern admitted a nonmatching string");
    const auto dot = Constraint::json_schema(R"({"type":"string","pattern":"^a.b$"})");
    for (const std::string separator : {"\n", "\r", "\u2028", "\u2029"})
        check(!accepts(compiler, dot, Json("a" + separator + "b").dump()),
              "ECMAScript dot admitted a line terminator");
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto compiler = make_compiler();
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            std::string line;
            while (std::getline(std::cin, line)) {
                Json response;
                try {
                    const auto input = Json::parse(line);
                    const auto constraint =
                        input.value("json_object", false)
                            ? Constraint::json_object()
                            : Constraint::json_schema(
                                  input.contains("schema_source")
                                      ? input["schema_source"].get<std::string>()
                                      : input.at("schema").dump());
                    auto session         = compiler.compile(constraint, {}, {});
                    response["accepted"] = Json::array();
                    for (const auto& text : input.at("candidates"))
                        response["accepted"].push_back(
                            accepts(compiler, constraint, text.get<std::string>()));
                } catch (const infernix::RequestError& error) {
                    response = {{"error", error.what()},
                                {"kind", static_cast<int>(error.kind())},
                                {"pointer", error.pointer()}};
                } catch (const std::exception& error) { response = {{"error", error.what()}}; }
                std::cout << response.dump() << std::endl;
            }
        } else {
            contracts(compiler);
            length_boundaries(compiler);
            dense_length_masks();
            std::cout << "JSON/schema CPU contracts passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
