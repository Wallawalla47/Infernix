#pragma once

// The Qwen4Exp forward pass (docs/maintainer/qwen3_8-flash-next-design.md §2, §8): embedding,
// the PLE injection, 48 hyper-connection blocks of GDN or QSA plus the offloaded MoE, the final
// mixer and the LM head. One call runs `batch` sequences of `width` consecutive positions: a
// prefill chunk is one sequence of many positions, a decode round several sequences of one.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/qsa.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// Program-owned recurrent state of every slot.
struct ForwardState {
    LinearAttentionStatePool* gdn = nullptr; // one layer per GDN block
    Tensor ple_conv;                         // BF16 [S*H, span, slots]
    std::vector<Tensor> qsa_tails;           // per attention block: BF16 [Di, R - 1, slots]
};

// Program-owned paged KV of the attention blocks.
struct ForwardKV {
    std::vector<ops::QsaKVLayer> layers; // per attention block
    Tensor block_tables;                 // I32 [pages per row, rows]
};

// Where each layer's routed experts live: device frames for resident experts, the pinned host
// bank otherwise.
struct ForwardExperts {
    const std::uint8_t* frame_base = nullptr;
    std::uint64_t frame_stride     = 0;
    std::vector<const std::int32_t*> frames; // per layer, device I32 [E]: frame or -1
    // Optional route log for the expert cache: layer l's routed ids (I32 [k, T]) are copied to
    // route_log + l * route_stride.
    std::int32_t* route_log  = nullptr;
    std::size_t route_stride = 0;
    // Device slots the MoE stages non-resident records into before computing them (shared by
    // every layer; see ops::MoeExpertSource).
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    // Per layer: the channel to the host expert engine for CPU-served misses (empty disables).
    std::vector<ops::MoeCpuChannel> cpu;
    // Prefill chunks stage their next pass of misses on this stream while one pass computes.
    cudaStream_t overlap_stream = nullptr;
    std::array<cudaEvent_t, 5> overlap_events{};
};

// A speculative verification call (design §11): `batch` sequences of `width` >= 2 positions. It
// leaves every recurrent state, PLE history and QSA tail unchanged and records what the commit of
// an accepted prefix needs instead; K/V and pooled keys are written as usual, since positions past
// the accepted prefix are rewritten before any later query reads them.
struct ForwardVerify {
    GdnReplayRecords gdn; // narrowed to the call's width; physical record row b = sequence b
    Tensor ple_inputs;    // BF16 [S*H, W, B]: the PLE convolution inputs
    Tensor qsa_keys;      // BF16 [Di, W, B, attention layers]: the raw index keys
};

struct ForwardBatch {
    Tensor ids;        // I32 [T]
    Tensor positions;  // I32 [T]
    Tensor ngram_rows; // U8 [row bytes, heads, T]: each column's FP8 n-gram rows
    Tensor slots;      // I32 [batch]: state slot of each sequence, updated in place
    Tensor table_rows; // I32 [batch]: KV block-table row of each sequence
    Tensor logit_columns; // I32 [n]: the columns whose logits are produced (each sequence's last
                          // for generation, every column for teacher-forced scoring)
    std::span<const std::int32_t> host_slots;      // the same slots on the host
    std::span<const std::int32_t> host_table_rows; // the same rows on the host
    std::int32_t batch = 0;
    std::int32_t width = 0;
    const ForwardVerify* verify = nullptr; // set for a speculative verification call
};

// Optional observation of every block, for reference comparison.
struct ForwardTap {
    std::vector<Tensor>* residuals = nullptr; // receives BF16 [S*H, T] copies, one per block
    std::vector<Tensor>* routes    = nullptr; // receives I32 [top k, T] routed expert ids, one per block
    // BF16 [H, T] copies per block of the mixer (GDN/QSA) input and output and the MoE input
    // and output, so each op chain can be checked against the reference on identical inputs.
    std::vector<Tensor>* mixer_inputs  = nullptr;
    std::vector<Tensor>* mixer_outputs = nullptr;
    std::vector<Tensor>* moe_inputs    = nullptr;
    std::vector<Tensor>* moe_outputs   = nullptr;
};

// The QSA geometry of the configuration.
[[nodiscard]] ops::QsaGeometry qsa_geometry(const TextConfig& config);

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
            ForwardState state, ForwardKV kv, ForwardExperts experts, std::int32_t max_context);

    // FP32 logits [V, n] of batch.logit_columns.
    void run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap = nullptr);

    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                     std::int32_t max_context);

private:
    void ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch);
    Tensor mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject);
    Tensor gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    Tensor attention(const AttentionParameters& p, const Tensor& x, std::uint32_t index,
                     const ForwardBatch& batch);
    Tensor moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap);

    const Parameters& parameters_;
    const TextConfig& config_;
    DeviceContext& device_;
    WorkspaceArena& work_;
    ForwardState state_;
    ForwardKV kv_;
    ForwardExperts experts_;
    std::int32_t max_context_;
    bool eager_chunk_ = false;
};

} // namespace ninfer::models::qwen4_exp::execution
