#pragma once

#include "text/byte_span.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The byte-level BPE tokenizers of the supported families (Hugging Face tokenizers' `tokenizer.json`): the
// vocabulary, merges and added tokens are data; the normalizer and the Split pre-tokenizer are one of these fixed
// pipelines, implemented by hand and checked against the resource's own description.
namespace infernix::text {

enum class TokenizerPipeline {
    // Qwen3.5/3.6/3.8: NFC, then Qwen's Split pattern (kQwenSplitPattern).
    Qwen,
    // Inkling 2: no normalizer, then the o200k-style Split pattern with case classes (kInklingSplitPattern).
    Inkling,
};

struct EncodeOptions {
    bool parse_added_tokens = true;
    std::size_t max_tokens  = std::numeric_limits<std::size_t>::max();
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct DecodedTokenView {
    std::string_view bytes;
    bool special = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
    // The family's pipeline: an omitted description in tokenizer.json takes it, a present one must equal it.
    TokenizerPipeline pipeline = TokenizerPipeline::Qwen;
};

struct BpeMergeRule {
    int rank   = 0;
    int result = -1;
};

struct BpeMergeEntry {
    std::uint64_t key = 0;
    int rank          = 0;
    int result        = -1;
};

// Merge table of the BPE encoder, open addressed with linear probing.
//
// Every adjacent symbol pair of every word is looked up here, and the lookup dominates
// Tokenizer::encode, so the table is stored flat: one power-of-two array of 16-byte entries
// instead of a bucket array plus one heap node per rule. A hit reads a single cache line
// rather than following a pointer chase, and the whole structure is smaller than the
// equivalent std::unordered_map even at a load factor below one half.
//
// The invariant that makes an empty slot recognisable is result < 0. Token ids come from
// parse_token_id, which rejects anything negative, so a stored rule always has result >= 0.
class BpeMergeTable {
public:
    // Sizes the table for rule_count rules. Capacity is the smallest power of two above
    // twice that count, so at least half of the slots stay empty and every probe chain ends.
    void reset(std::size_t rule_count) {
        std::size_t capacity = 1;
        while (capacity < rule_count * 2U) { capacity <<= 1U; }
        slots_.assign(capacity, BpeMergeEntry{});
        mask_ = capacity - 1U;
    }

    // Returns false when the key is already present, which is how the loader detects a
    // duplicate pair in model.merges.
    bool insert(std::uint64_t key, BpeMergeRule rule) {
        std::uint64_t index = mix(key) & mask_;
        while (slots_[index].result >= 0) {
            if (slots_[index].key == key) { return false; }
            index = (index + 1U) & mask_;
        }
        slots_[index] = BpeMergeEntry{.key = key, .rank = rule.rank, .result = rule.result};
        return true;
    }

    [[nodiscard]] const BpeMergeEntry* find(std::uint64_t key) const noexcept {
        std::uint64_t index = mix(key) & mask_;
        while (true) {
            const BpeMergeEntry& entry = slots_[index];
            if (entry.result < 0) { return nullptr; }
            if (entry.key == key) { return &entry; }
            index = (index + 1U) & mask_;
        }
    }

private:
    // splitmix64 finalizer. A key is two token ids packed into one word, so its low bits
    // alone would cluster badly across a power-of-two slot count.
    static std::uint64_t mix(std::uint64_t key) noexcept {
        key += 0x9e3779b97f4a7c15ULL;
        key = (key ^ (key >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27U)) * 0x94d049bb133111ebULL;
        return key ^ (key >> 31U);
    }

    // One empty slot by default, so find() on a default-constructed table terminates.
    std::vector<BpeMergeEntry> slots_ = std::vector<BpeMergeEntry>(1);
    std::uint64_t mask_               = 0;
};

struct TokenBoundaryResult {
    // exact_frontier is present when the byte marker is also a boundary in the complete token
    // stream. stable_frontier retains only a normalization- and pre-tokenization-complete prefix.
    std::optional<std::size_t> exact_frontier;
    std::size_t stable_frontier = 0;
};

struct BoundaryEncodedText {
    std::vector<int> input_ids;
    std::vector<TokenBoundaryResult> boundaries;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    // Literal spans suppress added-token recognition without splitting ordinary NFC/BPE runs.
    // `forced_splits` (ascending bytes) do split them: the ordinary text on each side of one is encoded
    // apart, as two separate encode calls would (a split inside an added token is ignored).
    BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           EncodeOptions options                         = {},
                           std::span<const text::ByteSpan> literal_spans = {},
                           std::span<const std::size_t> forced_splits    = {}) const;
    // The ids' bytes joined as the artifact's decoder spells them, then validated: a sequence that
    // is invalid or incomplete throws, so a caller never sees a partial character.
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    // The same bytes `decode` joins, without the validation: a caller that cuts a window between
    // two ids of one character receives that character's bytes as they are.
    [[nodiscard]] std::string decode_bytes(std::span<const int> ids, DecodeOptions options = {}) const;
    [[nodiscard]] DecodedTokenView decoded_token(int id) const;
    [[nodiscard]] std::string_view decode_token_bytes(int id,
                                                      bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept { return decoded_token_bytes_.size(); }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;
    // The added token whose content is exactly `content`, or null.
    [[nodiscard]] const AddedToken* find_added_token(std::string_view content) const noexcept;

    // The byte-level token of one raw byte, or -1 when the vocabulary has none.
    [[nodiscard]] int byte_token_id(std::uint8_t byte) const noexcept {
        return byte_token_ids_[byte];
    }

    // Transparent string hashing, so a pre-tokenized word (a view into the input) is looked up without a copy.
    struct WordHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view word) const noexcept { return std::hash<std::string_view>{}(word); }
    };
    using WordTable = std::unordered_map<std::string, int, WordHash, std::equal_to<>>;

private:
    TokenizerPipeline pipeline_ = TokenizerPipeline::Qwen;
    std::vector<std::string> decoded_token_bytes_;
    std::vector<bool> valid_token_ids_;
    std::vector<bool> special_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    BpeMergeTable bpe_merge_rules_;
    std::array<int, 256> byte_token_ids_{};
    // model.ignore_merges: a pre-tokenized word that is itself a vocabulary token (its raw bytes here) is emitted
    // as that token without merging. Empty when the model merges every word.
    WordTable whole_words_;
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace infernix::text
