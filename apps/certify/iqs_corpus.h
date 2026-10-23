#pragma once

// The IQS certification corpus (eval/corpora/iqs-1; docs/maintainer/model-quality.md §6): sets of
// documents, each scored as one window of a set's token length. A text document is tokenized raw; a chat
// document is rendered with the model's own chat template (its final assistant message continued, so
// every token is the model's own formatting). Decode arms force decode_tokens tokens at each of the set's
// decode starts of every window.

#include "infernix/engine.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace infernix::certify {

struct CorpusDocument {
    std::string id;
    std::string domain;
    std::string kind; // "text" or "chat"
    std::filesystem::path path;
};

struct CorpusSet {
    std::string name;
    std::uint32_t window_tokens = 0;
    std::uint32_t decode_tokens = 0;
    std::vector<std::uint32_t> decode_starts;
    std::vector<CorpusDocument> documents;
};

struct Corpus {
    std::string corpus_id;
    std::filesystem::path manifest;
    // SHA-free content fingerprint of the manifest and every document's bytes (FNV-1a, 64 bits).
    std::uint64_t content_hash = 0;
    CorpusSet set;
};

// Loads one set of a manifest and checks its documents exist.
[[nodiscard]] Corpus load_corpus(const std::filesystem::path& manifest, const std::string& set);

struct Window {
    std::uint32_t index = 0;
    std::string id;
    std::string domain;
    std::vector<TokenId> tokens; // exactly the set's window_tokens
    // generated[i]: token i is one the model writes (all of a text window; a chat's assistant turns only), so the
    // position that predicts it is scored. The rest of a chat (system prompt, tool schemas, user turns, tool
    // responses) is text the model never trains on: its predictions are arbitrary and numerically fragile.
    std::vector<std::uint8_t> generated;
    std::uint64_t input_hash = 0;
};

// Every document of the set as one window, tokenized by the Engine's artifact.
[[nodiscard]] std::vector<Window> load_windows(const Engine& engine, const Corpus& corpus);

} // namespace infernix::certify
