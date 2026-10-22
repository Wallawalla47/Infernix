#pragma once
#include "models/qwen3_5/program/program.h"

namespace infernix::models::qwen3_5::execution {
struct MtpCausalAttentionEnvelopes;
struct DFlashEnvelopes;
} // namespace infernix::models::qwen3_5::execution

namespace infernix::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
[[nodiscard]] std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t neural_drafts);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window);
// An adaptive-dflash verify-only family (SpeculativeRoundShape::split): the target alone, one
// topology per exact B across its resource tiers.
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_verify_graph_profiles(std::uint32_t capacity,
                                                                              std::uint32_t verify_drafts);

// `k` drafts are verified; the following MTP proposal runs `next_k` autoregressive steps.
[[nodiscard]] execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity,
                               std::uint32_t next_k);
// The drafter attention envelopes of a round over frontiers [min, max]. Under column-invariant
// decode every row drafts alone and must not see another row's context length in its attention
// route, so both envelopes are the context capacity, the bound the workspace is planned for.
[[nodiscard]] execution::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier,
                                                          std::uint32_t max_frontier,
                                                          bool column_invariant,
                                                          std::uint32_t capacity);

} // namespace infernix::models::qwen3_5::detail
