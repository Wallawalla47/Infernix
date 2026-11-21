// Inkling 2's prompt tokenization on the real tokenizer: the official chat template through Infernix's Jinja engine,
// tokenized with the render's literal spans (client text never becomes an added token), against Hugging Face
// (tests/fixtures/inkling2/chat_template_golden.json, generate_chat_template_golden.py):
// 1. Every conversation whose client text spells no marker gives Hugging Face's ids exactly.
// 2. A conversation whose client strings all spell markers (content, reasoning, an assistant's name, a tool's
//    description, call arguments, a tool result) holds exactly the added tokens of its twin with the spellings
//    replaced: Hugging Face's ids hold 14 more, its tokenizer recognizing added tokens anywhere in the text.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR  tokenizer.json, tokenizer_config.json and generation_config.json

#include "text/jinja.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::map<int, int> added_counts(const infernix::text::Tokenizer& tokenizer, const std::vector<int>& ids) {
    std::map<int, int> counts;
    for (const int id : ids) {
        if (tokenizer.is_special_token(id)) { ++counts[id]; }
    }
    return counts;
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_INKLING_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory);
        const infernix::text::Tokenizer tokenizer({.tokenizer_json         = read_file(root / "tokenizer.json"),
                                                   .tokenizer_config_json  = read_file(root / "tokenizer_config.json"),
                                                   .generation_config_json = read_file(root / "generation_config.json"),
                                                   .pipeline = infernix::text::TokenizerPipeline::Inkling});
        const std::filesystem::path source(INFERNIX_SOURCE_DIR);
        const infernix::text::JinjaTemplate tmpl(read_file(source / "tools" / "chat_templates" / "inkling2.jinja"),
                                                 "inkling2.jinja");
        const Json golden = Json::parse(read_file(source / "tests" / "fixtures" / "inkling2" / "chat_template_golden.json"));
        int failures = 0, parity = 0;
        std::map<std::string, std::vector<int>> ids_by_name;
        for (const Json& c : golden.at("cases")) {
            const std::string name = c.at("name").get<std::string>();
            const infernix::text::TemplateOutput rendered = tmpl.render(c.at("context"));
            const std::vector<int> ids =
                tokenizer.encode_with_boundaries(rendered.text, {}, {}, rendered.literal_spans).input_ids;
            ids_by_name[name] = ids;
            if (c.contains("spoof_of")) { continue; }
            ++parity;
            if (ids != c.at("ids").get<std::vector<int>>()) {
                ++failures;
                std::cerr << "FAIL " << name << ": ids differ from Hugging Face's (" << ids.size() << " vs "
                          << c.at("ids").size() << ")\n";
            }
        }
        for (const Json& c : golden.at("cases")) {
            if (!c.contains("spoof_of")) { continue; }
            const std::string name = c.at("name").get<std::string>();
            const auto spoofed     = added_counts(tokenizer, ids_by_name.at(name));
            const auto twin        = added_counts(tokenizer, ids_by_name.at(c.at("spoof_of").get<std::string>()));
            const auto hf          = added_counts(tokenizer, c.at("ids").get<std::vector<int>>());
            int total = 0, twin_total = 0, hf_total = 0;
            for (const auto& [id, n] : spoofed) { total += n; }
            for (const auto& [id, n] : twin) { twin_total += n; }
            for (const auto& [id, n] : hf) { hf_total += n; }
            std::cout << name << ": " << total << " added tokens (its twin " << twin_total << ", Hugging Face "
                      << hf_total << ")\n";
            if (spoofed != twin) {
                ++failures;
                std::cerr << "FAIL " << name << ": client text became added tokens\n";
            }
        }
        std::cout << parity << " conversations compared with Hugging Face's ids\n";
        if (failures != 0) {
            std::cerr << failures << " failures\n";
            return 1;
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
