#pragma once

// The --lossless route's Qwen3.5 text forward (docs/maintainer/lossless-route.md §2): Hugging Face's
// decoder operation by operation, each correctly rounded, over one causal-scoring window in one pass
// (attention over the window's own keys, GDN and convolution from zero state) or, for generation, over
// a call's columns continuing their sequences through the KV cache and the recurrent state slots.

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "infernix/types.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/offloaded_moe.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace infernix::models::qwen3_5::execution {

// The window's rotary table (lossless_rope_table) from its positions (I32 [T] or [T, 3]), allocated
// from `workspace` for the window's layers.
Tensor lossless_rope_window(const TextConfig& config, const Tensor& positions, WorkspaceArena& workspace,
                            cudaStream_t stream);
[[nodiscard]] std::size_t lossless_rope_window_bytes(const TextConfig& config, std::int32_t tokens);

// The cache form of a call (generation): its columns are `sequences` equal runs, each continuing its
// sequence. Attention stores a run's keys and values in the paged cache (the lossless bf16 profile) and
// reads the sequence's keys 0..position from it; the convolution and GDN continue from the run's source
// slot and leave their state in its destination slot. The window form (no cache) scores one window from
// zero state over its own keys.
struct LosslessCache {
    const Tensor* positions         = nullptr; // I32 [T]: each column's position
    const Tensor* table_rows        = nullptr; // I32 [B]: each run's block table row
    const Tensor* source_slots      = nullptr; // I32 [B]
    const Tensor* destination_slots = nullptr; // I32 [B]
    std::int32_t sequences          = 1;
    std::int32_t max_keys           = 0;       // bounds every position + 1
    std::int32_t key_capacity       = 0;       // the cache's context
    PagedKVBatchLayerView kv;                  // the layer's cache (attention layers)
    Tensor conv_states;                        // the layer's BF16 [C, K - 1, slots] (GDN layers)
    Tensor recurrent_states;                   // the layer's FP32 [128, 128, Hv, slots] (GDN layers)
};

// The design reference (docs/maintainer/lossless-route.md §3b): the configuration's declared lossy steps on
// the lossless route, in the window form. Inactive, the layer is --lossless's.
struct LosslessDesign {
    bool active       = false;                    // each projection input takes its Use's declared quantization
    KvCacheStorage kv = KvCacheStorage::BFloat16; // the KV format whose codec K and V cross
};

// A diagnostic tap on the residual stream: lossless_layer calls it with "mixer" after a layer's attention or
// delta-net block (the residual then holds x + mixer(norm(x))). Empty (the default) costs nothing; the
// Program sets one to dump the stream for a layer-by-layer comparison (INFERNIX_LOSSLESS_DUMP_DIR).
using ResidualTap = std::function<void(const char* tag, const Tensor& x, cudaStream_t stream)>;
void set_lossless_residual_tap(ResidualTap tap);

// One decoder layer over the call's residual x (BF16 [H, T]), in place; cache null: the window form. An
// offloaded MoE layer (OffloadedMoeParameters) reads its routed and shared experts from `experts` (the
// Program's), every expert on the GPU through the offloaded op's lossless pass; in a streamed pass
// (call->streamed) the expert stream's ring holds the layer's missing experts, copied once for the pass.
void lossless_layer(const BlockParameters& block, const TextConfig& config, MixerKind kind, const Tensor& rope_table,
                    Tensor& x, WorkspaceArena& workspace, cudaStream_t stream, const LosslessCache* cache = nullptr,
                    const LosslessDesign& design = {}, OffloadedExperts* experts = nullptr,
                    const OffloadedCall* call = nullptr);
// Its workspace; for the cache form `cache` gives sequences and key_capacity (tokens / sequences the run
// width).
[[nodiscard]] std::size_t lossless_layer_workspace_bytes(const BlockParameters& block, const TextConfig& config,
                                                         MixerKind kind, std::int32_t tokens,
                                                         const LosslessCache* cache   = nullptr,
                                                         const LosslessDesign& design = {});

// The final norm (Qwen3_5RMSNorm) into out (BF16 [H, T]).
void lossless_final_norm(const Tensor& x, const Tensor& weight, const TextConfig& config, Tensor& out,
                         cudaStream_t stream);

// The scoring logits: fp32(hidden W_head) (FP32 [vocab, T]); hidden BF16 [H, T].
void lossless_head(const Tensor& hidden, const Weight& head, Tensor& logits, WorkspaceArena& workspace,
                   cudaStream_t stream);
[[nodiscard]] std::size_t lossless_head_workspace_bytes(const Weight& head, std::int32_t tokens);

} // namespace infernix::models::qwen3_5::execution
