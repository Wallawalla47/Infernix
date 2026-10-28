// The Qwen tokenizer (frontend/tokenizer.cpp: its hand-written pre-tokenizer split and BPE) against
// Hugging Face `tokenizers` on a real tokenizer.json (candidates LE-10). The golden ids come from
// generate_tokenizer_golden.py (tests/fixtures/qwen3_5/tokenizer_golden.json) over about 2,600
// adversarial strings: CRLF and tab runs, NBSP and zero-width spaces, combining marks, ZWJ emoji, CJK
// and other scripts, number forms, apostrophes, paths, code and special-token spellings.
//
//   INFERNIX_TEST_TOKENIZER_DIR   a directory holding tokenizer.json, tokenizer_config.json and
//                                 generation_config.json (the generator writes it from an artifact)
//
// The golden belongs to one tokenizer.json (its byte count and SHA-256 are recorded); a directory
// holding another tokenizer skips, since its ids would not apply.

#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fi = infernix::models::qwen3_5::frontend;
using Json   = nlohmann::json;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory);
        const std::string tokenizer_json  = read_file(root / "tokenizer.json");
        const std::string tokenizer_config = read_file(root / "tokenizer_config.json");
        const std::string generation       = read_file(root / "generation_config.json");
        const Json golden =
            Json::parse(read_file(std::filesystem::path(INFERNIX_SOURCE_DIR) / "tests" / "fixtures" / "qwen3_5" /
                                  "tokenizer_golden.json"));
        if (golden.at("tokenizer_json_bytes").get<std::size_t>() != tokenizer_json.size()) {
            std::cout << "skip: the tokenizer.json is not the one the golden ids were made with\n";
            return 77;
        }
        const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                       .tokenizer_config_json  = tokenizer_config,
                                       .generation_config_json = generation});
        int failures = 0;
        std::size_t compared = 0;
        for (const Json& item : golden.at("cases")) {
            const std::string text = item.at("text").get<std::string>();
            const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
            const std::vector<int> actual   = tokenizer.encode(text);
            ++compared;
            if (actual == expected) { continue; }
            if (++failures <= 20) {
                std::cerr << "FAIL: ids differ for " << Json(text).dump() << "\n  expected " << Json(expected).dump()
                          << "\n  actual   " << Json(actual).dump() << '\n';
            }
        }
        std::cout << compared << " strings compared, " << failures << " differ\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
