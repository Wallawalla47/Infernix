#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <atomic>
#include <compare>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace infernix::runtime {

// A unit of execution failed in a way that leaves the engine usable (an expert record that could
// not be read, a host service that stopped answering): the engine fails the requests the unit
// served, as for a recoverable logic error, and keeps serving.
class RecoverableExecutionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// One request's own output cannot continue (its grammar refused a committed token). A commit
// releases that row like a cancellation and fails only its request; anywhere else it is handled as
// a RecoverableExecutionError.
class RequestOutputError : public RecoverableExecutionError {
public:
    using RecoverableExecutionError::RecoverableExecutionError;
};

// Why one row of a pending batch cannot commit: the Engine fails that row's request alone (the
// commit releases it as CommitDisposition::FailedReleased) and commits the other rows.
enum class PendingRowFailure : std::uint8_t {
    None,
    // The row's emitted tokens reach a position where its grammar admits no token.
    ConstraintDeadEnd,
    // The round's sampler found no finite logit to select from (ops::SamplingConfig::nonfinite):
    // the row's tokens are placeholders. The Program withdraws the prefix-cache entries the
    // request published (hybrid-prefix-cache-spec §7.8): its state may be non-finite too.
    NonFiniteLogits,
};

namespace testing {
// Fault injection for engine-level tests: the `rows`-th later row a Program checks for a failure (one
// per row of a sampled round, the prompt's first token included; process-wide) reports non-finite
// logits, as a NaN reaching the sampler would. 0 disarms.
inline std::atomic<std::uint32_t>& nonfinite_fault_rows() noexcept {
    static std::atomic<std::uint32_t> rows{0};
    return rows;
}
inline void set_nonfinite_fault(std::uint32_t rows) noexcept { nonfinite_fault_rows().store(rows); }
// Consumes one row check; true on the armed one.
[[nodiscard]] inline bool take_nonfinite_fault() noexcept {
    std::uint32_t left = nonfinite_fault_rows().load(std::memory_order_relaxed);
    while (left != 0) {
        if (nonfinite_fault_rows().compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}
} // namespace testing

// Borrowed for one synchronous Program call. Engine maps compact rows to request-owned matchers.
class TokenMaskProvider {
public:
    virtual ~TokenMaskProvider()                                           = default;
    [[nodiscard]] virtual bool constrained(std::size_t row) const noexcept = 0;
    // Returns dead-end position bits. Even dead/unreachable positions receive a safe nonempty
    // device mask. A dead end fails a row only if verification reaches that position.
    [[nodiscard]] virtual std::uint64_t fill(std::size_t row, std::span<const TokenId> drafts,
                                             std::span<std::uint32_t> words) = 0;
    virtual void uploaded(std::size_t row, std::size_t bytes) noexcept       = 0;
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
    FailedReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    bool failed                   = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
    // Set on the completing step of a request that named readout tokens.
    std::optional<PromptReadout> readout;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace infernix::runtime
