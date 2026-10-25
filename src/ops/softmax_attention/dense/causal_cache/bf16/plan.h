#pragma once

#include "infernix/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

namespace infernix::ops::detail {

struct Bf16KvCausalPlan {
    Bf16KvInstance instance;
    Bf16KvPartition partition;
    int query_heads;
    int width;
    int batch;
    CausalAttentionExecutionEnvelope envelope;
    // A tiled (prompt-route) plan's key splits per query tile: 1, or more under
    // envelope.bf16_prompt_split (bf16_kv_prompt_splits).
    int prompt_splits = 1;

    bool grouped() const { return bf16_kv_instance_description(instance).grouped; }
};

// The bf16-prompt-split count of a tiled launch: of 1-4 equal key shares per query tile, the one
// whose CTAs fill the last wave of the SMs best (a larger count only when it fills at least 3 % more),
// with every share at least 1,024 keys and its partials within the prompt split workspace.
int bf16_kv_prompt_splits(int query_heads, int width, int query_rows, std::uint32_t visible_keys,
                          std::size_t workspace_bytes, int multiprocessor_count);

Bf16KvCausalPlan make_bf16_kv_causal_plan(int query_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t bf16_kv_workspace_bytes(int query_heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

} // namespace infernix::ops::detail
