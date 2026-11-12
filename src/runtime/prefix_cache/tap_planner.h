#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::runtime::prefix_cache {

// Prompt boundary facts supplied by the Frontend (docs/maintainer/hybrid-prefix-cache-spec.md
// §7.1). Positions are exact token frontiers.
enum class TapHintKind : std::uint8_t {
    Explicit,         // client-named breakpoint (explicit-evidence protocol / PromptInput marker)
    GenerationOpener, // start of the final assistant generation opener
    Structural,       // end of tools, end of the leading System/Developer block
    // A protocol-automatic marker (OpenAI default prompt caching, Anthropic request-level
    // cache_control). Planned like a structural boundary, but it marks the conversation's latest
    // turn rather than a prefix conversations share, so its snapshot can be superseded.
    Automatic,
    MessageBoundary, // any message boundary; used to snap ladder taps
};

struct TapHint {
    std::uint32_t position = 0;
    TapHintKind kind       = TapHintKind::MessageBoundary;
};

// Span whose interior (begin, end) no resume point may lie in (a Vision item's tokens).
struct TapExclusion {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
};

// The exclusion of a media item covering tokens [begin, end). A snapshot's identity is its anchor
// path (full blocks, each carrying the media key of the items that begin before its end) plus the
// tokens of its partial tail block, so an item that begins inside the tail block before the
// frontier is in no key: two prompts that differ only by it would share the snapshot. The span
// therefore reaches at least to the end of the block the item begins in.
[[nodiscard]] TapExclusion media_exclusion(std::uint32_t begin, std::uint32_t end) noexcept;

// How the prefill loop realizes a tap.
enum class TapPlacement : std::uint8_t {
    // Split the prefill chunk at `position` and snapshot there: semantic boundaries where a
    // later prompt is expected to diverge.
    Exact,
    // Snapshot at a prefill chunk boundary instead of splitting: at the end of the chunk that
    // reaches `position`, or at the start of the prompt's final chunk when `position` lies in it.
    // A split costs one more forward pass, a chunk boundary costs nothing.
    Flexible,
};

struct PlannedTap {
    std::uint32_t position = 0;
    TapPlacement placement = TapPlacement::Exact;
    // A boundary later prompts may share across conversations (a client breakpoint or a
    // structural boundary): its snapshot is published as SnapshotKind::Boundary, superseded only
    // while no other conversation has continued from it.
    bool boundary = false;
    // Planned under `opener_near_base` (plan_taps): captured even when the lineage's deepest snapshot lies
    // less than kMinimumTapSeparation before it.
    bool forward = false;

    [[nodiscard]] friend bool operator==(PlannedTap, PlannedTap) noexcept = default;
};

// Two taps closer than this cover too little to pay for a second snapshot and split.
inline constexpr std::uint32_t kMinimumTapSeparation = 64;
// A forward snapshot (plan_taps `opener_near_base`) covers at least this many tokens: its split costs
// about as much as one short call re-reading 10-15 tokens (Flash-Next, 2026-11-11: a pair call +10..+14 ms
// against no split; a short call 0.9-1.8 ms a token).
inline constexpr std::uint32_t kMinimumForwardTokens = 16;

struct TapPlannerConfig {
    std::uint32_t max_new_taps   = 8;
    std::uint32_t ladder_tokens  = 4096;
    std::uint32_t min_gap_tokens = 1024;
};

// Returns taps sorted by position with base < position <= prompt_tokens - 1, outside every
// exclusion and not near an existing snapshot frontier on the matched path. Priority: Explicit,
// GenerationOpener, Structural, then the flexible prompt tail and geometric history ladder
// (prompt_tokens - G * 2^k, snapped back to a message boundary when one lies within the gap).
// A snapshot serves every prompt diverging after it, so proximity only makes a later tap redundant:
// semantic boundaries (opener, structural) closer than kMinimumTapSeparation keep only the
// earliest, and a non-Explicit, non-ladder candidate is dropped when an accepted tap, an existing
// snapshot or the base lies at most kMinimumTapSeparation before it. Ladder taps keep
// min_gap_tokens spacing on both sides. With `opener_near_base`, the generation opener is kept however
// close it is to the base (at least kMinimumForwardTokens past it) or to an existing snapshot at or before
// the base: the caller's evidence is that
// the next prompt resumes at this prompt's opener (the conversation does not echo its generated turns),
// so the snapshot moves forward instead of leaving the next turn to recompute this one.
[[nodiscard]] std::vector<PlannedTap> plan_taps(std::uint32_t prompt_tokens, std::uint32_t base,
                                                std::span<const TapHint> hints,
                                                std::span<const std::uint32_t> existing_frontiers,
                                                std::span<const TapExclusion> exclusions,
                                                const TapPlannerConfig& config, bool opener_near_base = false);

} // namespace infernix::runtime::prefix_cache
