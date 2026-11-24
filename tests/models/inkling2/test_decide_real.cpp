// POST /v1/decide's recipe on Inkling 2's real tokenizer and template (CPU only):
// 1. The label alphabet (single-token labels, as infernix-serve builds it) builds on this tokenizer, and every generated
//    primitive (number, scalar, point, box) is refused (question_type_unsupported): the tokenizer writes a number's
//    digits in groups ("10" is one token), never one digit a step as the primitives spell them.
// 2. A prompt that opens the answer (PromptOptions::open_answer, as /v1/decide prepares every question) ends with the
//    answer's opener `<|message_model|><|content_text|>`, counts the same tokens it prepares, opens no reasoning, and its
//    output session publishes the first generated token as content; an output constraint beside it is refused.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR  tokenizer.json, tokenizer_config.json and generation_config.json

#include "models/inkling2/frontend/frontend.h"
#include "serve/decide.h"
#include "text/tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ink = infernix::models::inkling2;
using infernix::TokenId;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_INKLING_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory), source(INFERNIX_SOURCE_DIR);
        const std::string tokenizer_json = read_file(root / "tokenizer.json");
        const std::string config         = read_file(root / "tokenizer_config.json");
        const std::string generation     = read_file(root / "generation_config.json");
        auto tokenizer = std::make_shared<const infernix::text::Tokenizer>(infernix::text::TokenizerResources{
            .tokenizer_json         = tokenizer_json,
            .tokenizer_config_json  = config,
            .generation_config_json = generation,
            .pipeline               = infernix::text::TokenizerPipeline::Inkling});
        const ink::FrontendResources resources{
            .tokenizer_json         = tokenizer_json,
            .tokenizer_config_json  = config,
            .chat_template_jinja    = read_file(source / "tools" / "chat_templates" / "inkling2.jinja"),
            .generation_config_json = generation,
            .tokenizer              = tokenizer};
        const ink::Frontend frontend(resources, {.max_context = 8192, .grammar_cache_bytes = 16ULL << 20});
        const infernix::serve::DecideEncoder encode = [&](std::string_view text) { return frontend.tokenize_text(text); };

        // 1. Labels and schedules.
        const std::vector<infernix::serve::DecideLabel> labels = infernix::serve::build_decide_alphabet(encode);
        std::cout << labels.size() << " single-token labels\n";
        check(labels.size() >= 62, "the alphabet holds at least A-Z, a-z and 0-9");
        using Kind = infernix::serve::DecideKind;
        for (const auto& [kind, name, digits] :
             std::vector<std::tuple<Kind, std::string, std::uint32_t>>{{Kind::Number, "number", 3},
                                                                       {Kind::Scalar, "scalar", 8},
                                                                       {Kind::Point, "point", 3},
                                                                       {Kind::Box, "box", 3}}) {
            infernix::serve::DecideQuestion question;
            question.kind         = kind;
            question.digits       = digits;
            question.instructions = "How many?";
            // The tokenizer writes digits in groups of up to three ("10" is one token), so a generated answer (one
            // digit a step) is refused before any prefill.
            std::string code;
            try {
                (void)infernix::serve::decide_schedule(question, encode);
            } catch (const infernix::serve::ApiException& error) { code = error.error().code; }
            check(code == "question_type_unsupported", name + ": a generated question is refused as unsupported");
        }

        // 2. An opened answer.
        infernix::PromptInput input;
        input.messages.push_back({.role = infernix::ChatRole::System, .parts = {{.text = "Answer with the letter."}}});
        input.messages.push_back({.role = infernix::ChatRole::User, .parts = {{.text = "Is water wet? A yes, B no"}}});
        input.options.enable_thinking = false;
        input.options.open_answer     = true;
        const auto id                 = [&](const char* s) { return static_cast<TokenId>(tokenizer->find_added_token(s)->id); };
        const ink::PreparedPrompt prompt = frontend.prepare(input);
        const auto ids                   = prompt.token_ids();
        check(ids.size() >= 2 && ids[ids.size() - 2] == id("<|message_model|>") && ids.back() == id("<|content_text|>"),
              "an opened answer's prompt ends with <|message_model|><|content_text|>");
        check(frontend.count_tokens(input) == ids.size(), "count_tokens counts the opener");
        check(!prompt.summary().opens_reasoning && !prompt.summary().starts_in_reasoning,
              "an opened answer opens no reasoning");
        infernix::PromptInput closed = input;
        closed.options.open_answer   = false;
        check(frontend.prepare(closed).token_ids().size() + 2 == ids.size(), "the opener adds two tokens to the prompt");
        check(frontend.prepare(closed).summary().opens_reasoning, "a closed prompt opens its reasoning in the output");
        ink::OutputSession session = frontend.make_output_session(prompt, {.include_model_defaults = false});
        const std::vector<TokenId> answer = frontend.tokenize_text("A");
        (void)session.preview_model(answer, 4, infernix::FinishReason::OutputLimit);
        std::string content;
        for (const auto& delta : session.commit_preview()) {
            if (delta.channel == infernix::OutputChannel::Content) { content += delta.text; }
        }
        check(content == "A", "the first generated token is the answer's content: " + content);
        bool refused = false;
        try {
            (void)frontend.make_output_session(prompt, {}, {}, {}, infernix::OutputConstraint::json_object());
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "an output constraint beside an opened answer is refused");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "inkling2 decide recipe: all checks passed\n";
    return 0;
}
