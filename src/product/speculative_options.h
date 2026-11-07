#pragma once

#include "infernix/types.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "off") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend (auto, off, mtp, dflash or dflash2): " + std::string(value));
}

// Applies a --spec value: "auto" lets the Engine choose the drafter from the artifact (backend None
// until it resolves it), "off" decodes plain, anything else names one.
inline void apply_speculative_backend(SpeculativeOptions& options, std::string_view value) {
    options.backend_auto = value == "auto";
    options.backend      = options.backend_auto ? SpeculativeBackend::None : parse_speculative_backend(value);
}

// The CLI's and the server's speculation without --spec: as --spec auto --draft-tokens auto, the drafter
// measured fastest among those the artifact can run (Qwen3.8-Flash-Next's DFlash2 drafter when --draft-model
// names it or its companion artifact lies beside the model), plain decode without one and on the lossless
// route. A fixed --draft-tokens or a --spec-min-p tunes one drafter, so it still needs --spec.
inline void apply_default_speculation(SpeculativeOptions& options) {
    if (options.draft_tokens != 0 && !options.draft_tokens_auto) {
        throw std::invalid_argument("--draft-tokens N tunes one drafter: it needs --spec mtp, dflash or dflash2 "
                                    "(without --spec, speculation is auto and takes --draft-tokens auto)");
    }
    if (options.draft_min_p) {
        throw std::invalid_argument("--spec-min-p tunes one drafter's cut: it needs --spec mtp or dflash2");
    }
    options.backend_auto      = true;
    options.draft_tokens_auto = true;
}

// A drafter is requested: a named backend, or --spec auto's.
[[nodiscard]] inline bool speculation_requested(const SpeculativeOptions& options) noexcept {
    return options.backend != SpeculativeBackend::None || options.backend_auto;
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

// Parses --spec-min-p: a draft-head probability in [0, 1].
[[nodiscard]] inline float parse_draft_min_p(std::string_view value) {
    const std::string text(value);
    char* end                = nullptr;
    const double probability = std::strtod(text.c_str(), &end);
    if (text.empty() || end != text.c_str() + text.size() || !(probability >= 0.0) ||
        !(probability <= 1.0)) {
        throw std::invalid_argument("--spec-min-p takes a probability in [0,1]: " + text);
    }
    return static_cast<float>(probability);
}

// "0.5"-style text of --spec-min-p, or "default" when unset.
[[nodiscard]] inline std::string draft_min_p_text(const SpeculativeOptions& options) {
    if (!options.draft_min_p) { return "default"; }
    char text[32];
    std::snprintf(text, sizeof(text), "%g", static_cast<double>(*options.draft_min_p));
    return text;
}

// Applies a --draft-tokens value: "auto" selects the automatic MTP draft length up to the model's
// most (draft_tokens 0 until the Engine resolves it), anything else a fixed count.
inline void apply_draft_tokens(SpeculativeOptions& options, std::string_view value) {
    options.draft_tokens_auto = value == "auto";
    options.draft_tokens      = 0;
    if (options.draft_tokens_auto) { return; }
    if (value.empty() || value.size() > 3 || value.find_first_not_of("0123456789") != std::string_view::npos) {
        throw std::invalid_argument("--draft-tokens takes a count or auto: " + std::string(value));
    }
    options.draft_tokens = static_cast<std::uint32_t>(std::stoul(std::string(value)));
}

// Applies a --ngram-draft-tokens value: "auto" lets the Engine choose the copy window and minimum
// match (ngram_draft_tokens 0 until it resolves them), anything else a fixed window (0: off).
inline void apply_ngram_draft_tokens(SpeculativeOptions& options, std::string_view value) {
    options.ngram_draft_tokens_auto = value == "auto";
    options.ngram_draft_tokens      = 0;
    if (options.ngram_draft_tokens_auto) { return; }
    if (value.empty() || value.size() > 3 || value.find_first_not_of("0123456789") != std::string_view::npos) {
        throw std::invalid_argument("--ngram-draft-tokens takes a count or auto: " + std::string(value));
    }
    options.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(std::string(value)));
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
        ((options.ngram_draft_tokens == 0 && !options.ngram_draft_tokens_auto) ||
         options.ngram_session_bytes < (1ULL << 20) ||
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
    if (options.ngram_draft_tokens_auto && !speculation_requested(options)) {
        throw std::invalid_argument("--ngram-draft-tokens auto requires --spec mtp, dflash, dflash2 or auto");
    }
    if (draft_tree_enabled(options)) {
        if (!speculation_requested(options)) {
            throw std::invalid_argument("--draft-tree-nodes requires --spec mtp, dflash, dflash2 or auto");
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
    // --spec auto chooses the drafter, so the draft length and the drafter's own cut follow it.
    if (options.backend_auto && !options.draft_tokens_auto) {
        throw std::invalid_argument("--spec auto chooses the drafter: it takes --draft-tokens auto");
    }
    if (options.backend_auto && options.draft_min_p) {
        throw std::invalid_argument("--spec-min-p tunes one drafter's cut: --spec auto takes none");
    }
    if (options.draft_min_p && options.backend != SpeculativeBackend::Mtp &&
        options.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument("--spec-min-p requires --spec mtp or dflash2");
    }
    // auto with dflash or dflash2: the block drafters' per-round verify width.
    if (options.draft_tokens_auto && !speculation_requested(options)) {
        throw std::invalid_argument("--draft-tokens auto requires --spec mtp, dflash, dflash2 or auto");
    }
    // With --spec auto it makes Qwen3.8-Flash-Next draft with DFlash2.
    if (!options.draft_model.empty() && options.backend != SpeculativeBackend::DFlash2 && !options.backend_auto) {
        throw std::invalid_argument("--draft-model is a DFlash2 drafter: it requires --spec dflash2 or auto");
    }
    if (options.draft_tree_paths < 2 || options.draft_tree_paths > 8) {
        throw std::invalid_argument("--draft-tree-paths must be in [2,8]");
    }
    if (options.ngram_tree_nodes != 0) {
        if (options.ngram_draft_tokens == 0 && !options.ngram_draft_tokens_auto) {
            throw std::invalid_argument("--ngram-tree-nodes requires ngram drafting: enable --ngram-draft-tokens");
        }
        if (options.ngram_tree_nodes < 4 || options.ngram_tree_nodes > 32) {
            throw std::invalid_argument("--ngram-tree-nodes must be 0 or in [4,32]");
        }
    }
    if (options.backend_auto) { return; }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument("--draft-tokens and --lm-head-draft need a drafter: --spec off takes neither");
        }
        return;
    case SpeculativeBackend::Mtp:
        // Qwen3.8-Flash-Next verifies up to 8 columns with column-invariant kernels (1..7); the
        // Qwen3.5/3.6 MTP planner accepts 1..5 itself. auto leaves the count to the Engine.
        if (options.draft_tokens_auto) { return; }
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,7] or auto");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens_auto) { return; }
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15] or auto");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens_auto) { return; }
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15] or auto");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace infernix::product
