// Video messages through the real Frontend against the reference processor's ids (candidates VK-03): the chat
// template's <|vision_start|><|video_pad|><|vision_end|> becomes one `<t seconds><|vision_start|>pads<|vision_end|>`
// group per temporal patch, with no outer pair, as Qwen3VLProcessor expands it up to transformers 5.9 (the
// Qwen3.5-family checkpoints' versions; from 5.10 it keeps the template's pair, which the golden stores for the
// record). tests/fixtures/qwen3_5/video_golden.json (generate_video_golden.py) holds lossless FFV1 videos of 2, 3, 9
// frames and 8 frames sampled to 4, and the whole rendered prompt's ids; each video goes through the Frontend's own
// FFmpeg decode, frame sampling, resize and template, and its prompt must equal the reference's id for id.
//
//   INFERNIX_TEST_VIDEO_RESOURCES_DIR  tokenizer.json, tokenizer_config.json, generation_config.json,
//                                      chat_template.jinja, preprocessor_config.json and
//                                      video_preprocessor_config.json of a Qwen3.5-family artifact (the generator
//                                      writes the directory)
//
// CPU only. Skipped (77) without the directory, or for a tokenizer.json other than the golden's.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/resources.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace q  = infernix::models::qwen3_5;
using infernix::TokenId;
using Json = nlohmann::json;

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::vector<std::uint8_t> decode_base64(const std::string& text) {
    static constexpr std::string_view kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::uint8_t> out;
    std::uint32_t buffer = 0;
    int bits             = 0;
    for (const char c : text) {
        if (c == '=') { break; }
        const std::size_t value = kAlphabet.find(c);
        if (value == std::string_view::npos) { throw std::runtime_error("invalid base64 in the golden"); }
        buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(buffer >> static_cast<unsigned>(bits)));
        }
    }
    return out;
}

std::string ids_text(std::span<const TokenId> ids, std::size_t from, std::size_t count) {
    std::string text = "[";
    for (std::size_t i = from; i < ids.size() && i < from + count; ++i) {
        text += (i == from ? "" : ",") + std::to_string(ids[i]);
    }
    return text + "]";
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_VIDEO_RESOURCES_DIR");
    if (directory == nullptr) {
        std::cout << "skip: set INFERNIX_TEST_VIDEO_RESOURCES_DIR\n";
        return 77;
    }
    try {
        const fs::path root(directory);
        const std::string tokenizer_json  = read_file(root / "tokenizer.json");
        const std::string tokenizer_cfg   = read_file(root / "tokenizer_config.json");
        const std::string generation_cfg  = read_file(root / "generation_config.json");
        const std::string chat_template   = read_file(root / "chat_template.jinja");
        const std::string preprocessor    = read_file(root / "preprocessor_config.json");
        const std::string video_processor = read_file(root / "video_preprocessor_config.json");
        const Json golden = Json::parse(
            read_file(fs::path(INFERNIX_SOURCE_DIR) / "tests" / "fixtures" / "qwen3_5" / "video_golden.json"));
        if (golden.at("tokenizer_json_bytes").get<std::size_t>() != tokenizer_json.size()) {
            std::cout << "skip: the tokenizer.json is not the one the golden ids were made with\n";
            return 77;
        }

        const auto tokenizer = std::make_shared<const infernix::text::Tokenizer>(
            infernix::text::TokenizerResources{tokenizer_json, tokenizer_cfg, generation_cfg});
        q::FrontendResources resources{.tokenizer_json                 = tokenizer_json,
                                       .tokenizer_config_json          = tokenizer_cfg,
                                       .chat_template_jinja            = chat_template,
                                       .generation_config_json         = generation_cfg,
                                       .preprocessor_config_json       = preprocessor,
                                       .video_preprocessor_config_json = video_processor};
        resources.tokenizer          = tokenizer;
        resources.public_token_count = static_cast<std::uint32_t>(tokenizer->vocab_size());
        q::FrontendOptions options;
        options.vision_enabled = true;
        options.max_context    = std::numeric_limits<std::uint32_t>::max();
        const q::Frontend frontend = q::make_frontend(resources, options);

        int failures = 0;
        for (const Json& item : golden.at("cases")) {
            const std::string name = item.at("name").get<std::string>();
            infernix::MessagePart video;
            video.kind              = infernix::MessagePartKind::Media;
            video.media.kind        = infernix::MediaKind::Video;
            video.media.bytes       = decode_base64(item.at("video_mkv_base64").get<std::string>());
            video.media.media_type  = "video/x-matroska";
            video.media.source_name = "golden.mkv";
            infernix::ChatMessage message;
            message.role = infernix::ChatRole::User;
            message.parts.push_back(std::move(video));
            message.parts.push_back(infernix::MessagePart{
                .kind = infernix::MessagePartKind::Text, .text = golden.at("text").get<std::string>(), .media = {}});
            infernix::PromptInput input;
            input.messages.push_back(std::move(message));

            const q::PreparedPrompt prepared = frontend.prepare(std::move(input));
            const std::span<const TokenId> ids = prepared.token_ids();
            const std::vector<TokenId> expected = item.at("ids").get<std::vector<TokenId>>();
            std::size_t at = 0;
            while (at < ids.size() && at < expected.size() && ids[at] == expected[at]) { ++at; }
            if (at == ids.size() && at == expected.size()) {
                std::cout << "PASS " << name << ": " << ids.size() << " ids\n";
                continue;
            }
            ++failures;
            std::cerr << "FAIL " << name << ": " << ids.size() << " ids against the reference's " << expected.size()
                      << "; first difference at " << at << ": " << ids_text(ids, at, 12) << " against "
                      << ids_text(expected, at, 12) << '\n';
            const auto& view = q::PreparedPromptAccess::view(prepared);
            if (!view.vision_items.empty()) {
                const auto& v = view.vision_items.front();
                std::cerr << "  Frontend grid " << v.grid.temporal << "x" << v.grid.height << "x" << v.grid.width
                          << " against the reference's " << item.at("grid").dump() << "; timestamps";
                for (const double t : v.timestamps) { std::cerr << ' ' << t; }
                std::cerr << "; the reference sampled " << item.at("indices").dump() << '\n';
            }
        }
        if (failures != 0) {
            std::cerr << failures << " of " << golden.at("cases").size() << " video prompts differ from the reference\n";
            return 1;
        }
        std::cout << "PASS: " << golden.at("cases").size() << " video prompts equal the reference id for id\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
