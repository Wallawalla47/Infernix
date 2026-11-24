#pragma once

// Inkling 2's frontend (design docs/maintainer/inkling2_mini-design.md §17): the official chat template through
// Infernix's Jinja engine, tokenized with the render's literal spans (client text never becomes an added token), and
// the output session (output_session.h) of each turn.
//
// This slice renders conversations for a new assistant turn: system/developer, user, assistant (with reasoning and
// tool calls) and tool turns, images in user and tool turns (with --vision: each image part's one placeholder expanded
// to its cells' image tokens; image.h processes the pixels), declared tools and the template's thinking effort
// (PromptOptions::reasoning_effort, enable_thinking false = "none", or a chat_template_kwargs reasoning_effort string
// or number in [0, 0.99]). A turn's output may be constrained (constraints.h): a content constraint, a tool policy
// (required, named, strict, parallel false or ToolConstraintMode::Basic's protection of every call), or both. It
// refuses what later milestones add: video, continuing a final assistant message, the next-turn history prefill and the
// reasoning-loop guard.

#include "infernix/types.h"
#include "models/inkling2/frontend/image.h"
#include "models/inkling2/frontend/output_session.h"
#include "models/inkling2/model.h"
#include "models/inkling2/ngram.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::inkling2 {

struct FrontendOptions {
    std::uint32_t max_context = 2'048;
    // Thinking text injected before the thinking close when a request reaches its thinking budget; empty injects the
    // close alone.
    std::string thinking_budget_message;
    // --vision: the tower's config (image input accepted) and the image-token budget of an image (--vision-max-merged).
    std::optional<VisionConfig> vision;
    std::uint32_t vision_max_cells = 32768;
    // The constraint compiler's cache of compiled grammars (EngineOptions::grammar_cache_bytes).
    std::size_t grammar_cache_bytes = 0;
};

// One image of a prompt: its image tokens [first_token, first_token + cells) and its processed patches.
struct PromptImage {
    std::uint32_t first_token = 0;
    ProcessedImage image;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::shared_ptr<const ToolNames> tools; // null: the prompt declares no tools
    std::vector<PromptImage> images;        // in prompt order
    // Prefix-cache lookup keys over token_ids (runtime/prefix_cache/block_hash.h): one chained hash per full 64-token
    // block and, with images, one cumulative image key per block (every image that begins before the block's end).
    std::vector<std::uint64_t> block_hashes;
    std::vector<std::uint64_t> block_extras;
    // Where the prompt's structure suggests snapshots (runtime/prefix_cache/tap_planner.h): every message's start, the
    // leading system block's last message and its end, and the generation opener.
    std::vector<runtime::prefix_cache::TapHint> tap_hints;
    // False: the prompt neither reuses nor publishes cached state (Engine::prepare_tokens without a prefix identity).
    bool reusable = true;
    // The prompt ends with the answer's opener (PromptOptions::open_answer): the reply starts inside its answer.
    bool answer_opened = false;
    PromptPreparationStats stats;
};

class PreparedPrompt {
public:
    PreparedPrompt() noexcept = default;
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept : data_(std::move(data)) {}

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }
    [[nodiscard]] const PreparedPromptData& data() const { return *data_; }
    // No n-gram archive (ngram.h): nothing to bind.
    [[nodiscard]] std::unique_ptr<NgramArchive::Request> bind_ngram(NgramArchive&, const NgramSessionHints&) const {
        return nullptr;
    }
    // The prompt's data, moved out (the Program's plan takes it).
    [[nodiscard]] std::unique_ptr<PreparedPromptData> take() && noexcept { return std::move(data_); }

private:
    std::unique_ptr<PreparedPromptData> data_;
};

class Frontend {
public:
    Frontend(const FrontendResources& resources, FrontendOptions options);

    [[nodiscard]] PreparedPrompt prepare(PromptInput input, const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input, const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids, bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string detokenize(std::span<const TokenId> token_ids) const;
    [[nodiscard]] std::string detokenize_bytes(std::span<const TokenId> token_ids) const;
    // The chat's ids and which of them the model writes: every assistant message, from its first <|message_model|>
    // through its <|content_model_end_sampling|> (the template's generation blocks; no other role writes either).
    [[nodiscard]] ChatTokens tokenize_chat(PromptInput input) const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                                                    const OutputOptions& output                       = {},
                                                    const ThinkingControlOptions& thinking            = {},
                                                    const std::optional<OutputConstraint>& constraint = {},
                                                    const ToolChoice& tool_choice                     = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    // The template renders a system turn at any position (developer turns render as system turns).
    [[nodiscard]] ChatTemplateRoleSupport chat_template_role_support() const noexcept { return {}; }
    [[nodiscard]] VisionImageBudget vision_image_budget() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const noexcept { return {}; }
    // Builds the constraint compiler's vocabulary index (the Engine runs it beside startup).
    void warm_constraints() const;

    // The rendered prompt text and its token ids (tests and tools).
    struct Rendered {
        std::string text;
        std::vector<TokenId> ids;
    };
    [[nodiscard]] Rendered render(const PromptInput& input, std::size_t max_tokens) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace infernix::models::inkling2
