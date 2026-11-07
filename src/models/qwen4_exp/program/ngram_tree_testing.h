#pragma once

// Qwen3.8-Flash-Next chooses a round's copy length, and whether to verify an n-gram tree, by expected tokens per
// measured round time (ProgramImpl::copy_choice and ngram_tree_for), so on an offloaded artifact copies are short
// and trees rare. This seam lets the n-gram tree real test cover the tree rounds whatever the costs.

namespace infernix::models::qwen4_exp::testing {

// With the override (true), a one-row round takes every copy longer than its drafter's drafts, to a two-path
// tree's depth, and verifies every eligible tree; without it (false, the default), the cost policy chooses. For
// the Programs' rounds after the call. For tests only.
void set_ngram_policy_override(bool override_policy) noexcept;
[[nodiscard]] bool ngram_policy_override() noexcept;

} // namespace infernix::models::qwen4_exp::testing
