#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// Parses --draft-tree-nodes: one to eight comma-separated column counts, entry c-1 applying to
// rounds of c rows and the last entry to every larger batch; 0 keeps a batch size on chain
// verification.
[[nodiscard]] inline std::array<std::uint32_t, kMaximumConcurrency>
parse_draft_tree_nodes(std::string_view value) {
    std::array<std::uint32_t, kMaximumConcurrency> nodes{};
    std::size_t count = 0;
    while (true) {
        const std::size_t comma     = value.find(',');
        const std::string_view item = value.substr(0, comma);
        if (item.empty() || item.size() > 3 || count == nodes.size() ||
            item.find_first_not_of("0123456789") != std::string_view::npos) {
            throw std::invalid_argument(
                "--draft-tree-nodes takes 1 to 8 comma-separated column counts");
        }
        nodes[count++] = static_cast<std::uint32_t>(std::stoul(std::string(item)));
        if (comma == std::string_view::npos) { break; }
        value.remove_prefix(comma + 1);
    }
    for (std::size_t c = count; c < nodes.size(); ++c) { nodes[c] = nodes[count - 1]; }
    return nodes;
}

// Applies a --draft-tree-nodes value: "auto" selects automatic tree widths, anything else is a
// per-batch-size table (parse_draft_tree_nodes).
inline void apply_draft_tree_nodes(SpeculativeOptions& options, std::string_view value) {
    options.draft_tree_auto  = value == "auto";
    options.draft_tree_nodes = {};
    if (!options.draft_tree_auto) { options.draft_tree_nodes = parse_draft_tree_nodes(value); }
}

[[nodiscard]] inline bool draft_tree_enabled(const SpeculativeOptions& options) noexcept {
    return options.draft_tree_auto ||
           std::any_of(options.draft_tree_nodes.begin(), options.draft_tree_nodes.end(),
                       [](std::uint32_t nodes) { return nodes != 0; });
}

// "auto", "16,16,12,12,0,0,0,0" style text of the per-batch-size tree table, or "off".
[[nodiscard]] inline std::string draft_tree_nodes_text(const SpeculativeOptions& options) {
    if (options.draft_tree_auto) { return "auto"; }
    if (!draft_tree_enabled(options)) { return "off"; }
    std::string text;
    for (std::size_t c = 0; c < options.draft_tree_nodes.size(); ++c) {
        if (c != 0) { text += ','; }
        text += std::to_string(options.draft_tree_nodes[c]);
    }
    return text;
}

// Widest neural verification: the draft tokens, or a DFlash2 tree's columns less the anchor.
[[nodiscard]] inline std::uint32_t
speculative_verify_drafts(const SpeculativeOptions& options) noexcept {
    std::uint32_t drafts = options.draft_tokens;
    for (const std::uint32_t nodes : options.draft_tree_nodes) {
        if (nodes != 0) { drafts = std::max(drafts, nodes - 1U); }
    }
    if (options.draft_tree_auto) {
        for (const std::uint32_t nodes : draft_tree_auto_widths(options.draft_tokens)) {
            drafts = std::max(drafts, nodes - 1U);
        }
    }
    return drafts;
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (options.ngram_archive_bytes != 0 &&
        (options.ngram_draft_tokens == 0 || options.ngram_session_bytes < (1ULL << 20) ||
         options.ngram_session_bytes > options.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram archive requires ngram drafting and session capacity "
                                    "between 1 MiB and total archive capacity");
    }
    // Which drafter combinations a model supports is the model Program's decision (Qwen3.5
    // verifies copy proposals only beside a neural drafter; Qwen4Exp also alone).
    if (options.ngram_draft_tokens != 0 &&
        (options.ngram_draft_tokens > 63 || options.ngram_min_match < 4 ||
         options.ngram_min_match > 64)) {
        throw std::invalid_argument("ngram drafts must be 1..63 with a minimum match of 4..64");
    }
    if (draft_tree_enabled(options)) {
        if (options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("--draft-tree-nodes requires --spec dflash2");
        }
        if (options.draft_tree_auto &&
            std::any_of(options.draft_tree_nodes.begin(), options.draft_tree_nodes.end(),
                        [](std::uint32_t nodes) { return nodes != 0; })) {
            throw std::invalid_argument("automatic tree widths take no --draft-tree-nodes table");
        }
        for (const std::uint32_t nodes : options.draft_tree_nodes) {
            if (nodes != 0 && (nodes < options.draft_tokens + 2U || nodes > 32U)) {
                throw std::invalid_argument(
                    "--draft-tree-nodes entries must be 0 or in [draft tokens + 2, 32]");
            }
        }
    }
    if (options.draft_tree_paths < 2 || options.draft_tree_paths > 8) {
        throw std::invalid_argument("--draft-tree-paths must be in [2,8]");
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
