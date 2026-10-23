#include "iqs_corpus.h"

#include "distribution.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace infernix::certify {
namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

std::uint64_t fnv(std::uint64_t hash, const std::string& bytes) {
    for (const unsigned char c : bytes) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

ChatRole role_of(const std::string& name) {
    if (name == "system") { return ChatRole::System; }
    if (name == "developer") { return ChatRole::Developer; }
    if (name == "user") { return ChatRole::User; }
    if (name == "assistant") { return ChatRole::Assistant; }
    if (name == "tool") { return ChatRole::Tool; }
    throw std::runtime_error("unknown chat role " + name);
}

// A chat document: {"tools": [json schemas, optional], "enable_thinking": bool (optional),
// "messages": [{"role", "content", "reasoning" (optional), "tool_calls": [{"id", "name",
// "arguments"}] (optional), "tool_call_id" (optional)}]}; its last message is the assistant's.
PromptInput chat_input(const std::string& text, const std::filesystem::path& path) {
    const nlohmann::json doc = nlohmann::json::parse(text);
    PromptInput input;
    for (const auto& message : doc.at("messages")) {
        ChatMessage out;
        out.role = role_of(message.at("role").get<std::string>());
        out.parts.push_back(MessagePart{.kind  = MessagePartKind::Text,
                                        .text  = message.value("content", std::string{}),
                                        .media = {}});
        out.reasoning_content = message.value("reasoning", std::string{});
        out.tool_call_id      = message.value("tool_call_id", std::string{});
        if (message.contains("tool_calls")) {
            for (const auto& call : message.at("tool_calls")) {
                out.tool_calls.push_back(ToolCall{.id             = call.value("id", std::string{}),
                                                  .name           = call.at("name").get<std::string>(),
                                                  .arguments_json = call.at("arguments").dump()});
            }
        }
        input.messages.push_back(std::move(out));
    }
    if (input.messages.empty() || input.messages.back().role != ChatRole::Assistant) {
        throw std::runtime_error(path.string() + ": a chat document ends with the assistant's message");
    }
    if (doc.contains("tools")) {
        for (const auto& tool : doc.at("tools")) { input.options.tool_jsons.push_back(tool.dump()); }
    }
    // The window is a prefix of the rendering, so the final assistant message renders complete (a continuation
    // needs thinking disabled, which a reasoning chat is not) and the new-turn prompt after it lies beyond the cut.
    if (doc.contains("enable_thinking")) { input.options.enable_thinking = doc.at("enable_thinking").get<bool>(); }
    // Reasoning kept in every assistant turn, not only the last (a reasoning chat's own text).
    if (doc.contains("preserve_thinking")) { input.options.preserve_thinking = doc.at("preserve_thinking").get<bool>(); }
    return input;
}

} // namespace

Corpus load_corpus(const std::filesystem::path& manifest, const std::string& set_name) {
    const std::string manifest_text = read_file(manifest);
    const nlohmann::json root       = nlohmann::json::parse(manifest_text);
    Corpus out;
    out.corpus_id = root.at("corpus_id").get<std::string>();
    out.manifest  = std::filesystem::absolute(manifest);
    const auto& sets = root.at("sets");
    if (!sets.contains(set_name)) { throw std::runtime_error(manifest.string() + " has no set " + set_name); }
    const auto& set           = sets.at(set_name);
    out.set.name              = set_name;
    out.set.window_tokens     = set.at("window_tokens").get<std::uint32_t>();
    out.set.decode_tokens     = set.at("decode_tokens").get<std::uint32_t>();
    out.set.decode_starts     = set.at("decode_starts").get<std::vector<std::uint32_t>>();
    for (const std::uint32_t start : out.set.decode_starts) {
        if (start == 0 || start + out.set.decode_tokens > out.set.window_tokens) {
            throw std::runtime_error("decode start " + std::to_string(start) + " leaves no room in the window");
        }
    }
    std::uint64_t hash = fnv(14695981039346656037ULL, manifest_text);
    const std::filesystem::path base = out.manifest.parent_path();
    for (const auto& doc : set.at("documents")) {
        CorpusDocument document{.id     = doc.at("id").get<std::string>(),
                                .domain = doc.at("domain").get<std::string>(),
                                .kind   = doc.value("kind", std::string("text")),
                                .path   = base / doc.at("path").get<std::string>()};
        if (document.kind != "text" && document.kind != "chat") {
            throw std::runtime_error(document.id + ": kind must be text or chat");
        }
        hash = fnv(hash, read_file(document.path));
        out.set.documents.push_back(std::move(document));
    }
    out.content_hash = hash;
    return out;
}

std::vector<Window> load_windows(const Engine& engine, const Corpus& corpus) {
    std::vector<Window> windows;
    for (std::size_t i = 0; i < corpus.set.documents.size(); ++i) {
        const CorpusDocument& document = corpus.set.documents[i];
        const std::string text         = read_file(document.path);
        std::vector<TokenId> tokens;
        std::vector<std::uint8_t> generated;
        if (document.kind == "text") {
            tokens = engine.tokenize_text(text);
            generated.assign(tokens.size(), 1);
        } else {
            ChatTokens chat = engine.tokenize_chat(chat_input(text, document.path));
            tokens          = std::move(chat.ids);
            generated       = std::move(chat.generated);
        }
        if (tokens.size() < corpus.set.window_tokens) {
            throw std::runtime_error(document.id + " holds " + std::to_string(tokens.size()) + " tokens; the " +
                                     corpus.set.name + " set's windows need " +
                                     std::to_string(corpus.set.window_tokens));
        }
        tokens.resize(corpus.set.window_tokens);
        generated.resize(corpus.set.window_tokens);
        Window window{.index = static_cast<std::uint32_t>(i), .id = document.id, .domain = document.domain,
                      .tokens = std::move(tokens), .generated = std::move(generated)};
        window.input_hash = perplexity::token_fingerprint(window.tokens);
        windows.push_back(std::move(window));
    }
    return windows;
}

} // namespace infernix::certify
