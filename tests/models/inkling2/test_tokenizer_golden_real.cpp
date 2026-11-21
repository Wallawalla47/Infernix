// The Inkling 2 tokenizer (text/tokenizer.cpp's Inkling pipeline: no normalizer, the hand-written o200k-style Split,
// BPE with ignore_merges) against Hugging Face `tokenizers` on the real tokenizer.json. The golden ids come from
// generate_tokenizer_golden.py (tests/fixtures/inkling2/tokenizer_golden.json): the shared adversarial corpus plus the
// letter-case runs, folded contractions, digit and slash runs the pattern distinguishes. Every case also decodes back
// to its text.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR     a directory holding tokenizer.json, tokenizer_config.json and
//                                           generation_config.json (the generator writes it)
//   INFERNIX_TEST_INKLING_TOKENIZER_GOLDEN  optional: another golden file of the same tokenizer (the generator's
//                                           --sweep over every codepoint) in place of the fixture
//
// The golden belongs to one tokenizer.json (its byte count is recorded); another tokenizer skips.

#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
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
        const std::string tokenizer_json   = read_file(root / "tokenizer.json");
        const std::string tokenizer_config = read_file(root / "tokenizer_config.json");
        const std::string generation       = read_file(root / "generation_config.json");
        const char* override_path          = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_GOLDEN");
        const std::filesystem::path golden_path =
            override_path != nullptr && *override_path != '\0'
                ? std::filesystem::path(override_path)
                : std::filesystem::path(INFERNIX_SOURCE_DIR) / "tests" / "fixtures" / "inkling2" / "tokenizer_golden.json";
        const Json golden = Json::parse(read_file(golden_path));
        if (golden.at("tokenizer_json_bytes").get<std::size_t>() != tokenizer_json.size()) {
            std::cout << "skip: the tokenizer.json is not the one the golden ids were made with\n";
            return 77;
        }
        const infernix::text::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                                   .tokenizer_config_json  = tokenizer_config,
                                                   .generation_config_json = generation,
                                                   .pipeline               = infernix::text::TokenizerPipeline::Inkling});
        int failures         = 0;
        std::size_t compared = 0;
        for (const Json& item : golden.at("cases")) {
            const std::string text          = item.at("text").get<std::string>();
            const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
            const std::vector<int> actual   = tokenizer.encode(text);
            ++compared;
            const bool round_trip = tokenizer.decode(actual) == text;
            if (actual == expected && round_trip) { continue; }
            if (++failures <= 20) {
                std::cerr << "FAIL: " << (actual == expected ? "decode differs" : "ids differ") << " for "
                          << Json(text).dump() << "\n  expected " << Json(expected).dump() << "\n  actual   "
                          << Json(actual).dump() << '\n';
            }
        }
        std::cout << compared << " strings compared, " << failures << " differ\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
