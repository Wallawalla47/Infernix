// Inkling 2's official chat template (tools/chat_templates/inkling2.jinja, byte-identical to the checkpoint's
// chat_template.jinja) through Infernix's Jinja engine against Hugging Face transformers' rendering
// (tests/fixtures/inkling2/chat_template_golden.json, written by generate_chat_template_golden.py): every case's text
// must be identical. The cases cover the thinking-effort line (default, strings, numbers, zero), system-first
// conversations, thinking turns, content parts, tool declarations with sorted compact JSON, tool calls with
// case-sensitively sorted arguments, tool results by id and by name, named assistants and non-ASCII text.

#include "text/jinja.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

int main() {
    try {
        const std::string source = read(INFERNIX_SOURCE_DIR "/tools/chat_templates/inkling2.jinja");
        const auto golden = nlohmann::ordered_json::parse(read(INFERNIX_SOURCE_DIR "/tests/fixtures/inkling2/chat_template_golden.json"));
        const infernix::text::JinjaTemplate tmpl(source, "inkling2.jinja");
        int failures = 0, cases = 0;
        for (const auto& c : golden.at("cases")) {
            ++cases;
            const std::string expected = c.at("expected").get<std::string>();
            std::string got;
            try {
                got = tmpl.render(c.at("context")).text;
            } catch (const std::exception& e) {
                got = std::string("<render failed: ") + e.what() + ">";
            }
            if (got != expected) {
                ++failures;
                std::size_t at = 0;
                while (at < got.size() && at < expected.size() && got[at] == expected[at]) { ++at; }
                std::cerr << "FAIL " << c.at("name").get<std::string>() << ": first difference at byte " << at
                          << "\n  expected: ..." << expected.substr(at > 40 ? at - 40 : 0, 120)
                          << "\n  got:      ..." << got.substr(at > 40 ? at - 40 : 0, 120) << '\n';
            }
        }
        std::cout << cases - failures << " of " << cases << " cases render as transformers does\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
