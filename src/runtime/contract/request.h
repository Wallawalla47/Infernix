#pragma once

#include "infernix/types.h"
#include <atomic>
#include <cstdint>
#include <optional>

namespace infernix::runtime {

// Engine has already selected the model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    std::vector<TokenId> readout_tokens;
    TokenConstraint constraint;
    // The Engine's idle-time pass over a finished turn (hybrid-prefix-cache-spec §7.9), which no client
    // waits for: it yields to a request at its next unit boundary, so a Program keeps its units short.
    bool background = false;
    // Whether the finished request publishes its endpoint snapshot (the resume point of the conversation's
    // next turn). False when its finished state is no conversation's next turn (a decision, set by the
    // Engine): the snapshot would only take a Device snapshot slot from a conversation that has one.
    bool publish_endpoint = true;
};

// A row whose thinking is open at a call's first position: its end-of-turn tokens, and the tool-call opener when it
// declares tools (withheld), take no probability until it has written the close token (ops::SamplingConfig::
// reasoning_close; the output session's reasoning_ban). close -1: none; withheld -1: none.
struct ReasoningBan {
    TokenId close    = -1;
    TokenId ends[2]  = {-1, -1};
    TokenId withheld = -1;
    [[nodiscard]] bool active() const noexcept { return close >= 0; }
};

struct ResolvedRequestOptions {
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
    // A decision (ExecutionOptions::decision): a scheduling property the Engine reads (RequestRecord::
    // is_decision); the Program sees only its consequence, execution.publish_endpoint.
    bool decision = false;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;
    // Empty or one token-aligned frontier within the accepted span where model-history
    // reconstruction gains an execution split. Frontend owns detection; Engine only transports
    // this relative position.
    std::optional<std::uint32_t> prefix_execution_split_after;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    bool publish_continuation             = true;
    // ResolvedExecutionOptions::publish_endpoint, as the Program plans it. Qwen3.5's Program reads it;
    // Inkling 2's publishes no endpoint at a finish anyway, and Qwen3.8-Flash-Next's (whose snapshots
    // live in its Host tier, no Device slot) publishes one regardless.
    bool publish_endpoint                 = true;
};

} // namespace infernix::runtime
