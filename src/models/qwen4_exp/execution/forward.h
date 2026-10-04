#pragma once

// The Qwen4Exp forward pass (docs/maintainer/qwen3_8-flash-next-design.md §2, §8): embedding,
// the PLE injection, 48 hyper-connection blocks of GDN or QSA plus the offloaded MoE, the final
// mixer and the LM head. One call runs `batch` sequences of `width` consecutive positions: a
// prefill chunk is one sequence of many positions, a decode round several sequences of one.

#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/qsa.h"

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
};

struct ForwardBatch {
    Tensor ids;        // I32 [T]
    Tensor positions;  // I32 [T]
    Tensor ngram_rows; // U8 [row bytes, heads, T]: each column's FP8 n-gram rows
    Tensor slots;      // I32 [batch]: state slot of each sequence, updated in place
    Tensor table_rows; // I32 [batch]: KV block-table row of each sequence
    Tensor last_columns; // I32 [batch]: the column whose logits each sequence produces
    std::span<const std::int32_t> host_slots;      // the same slots on the host
    std::span<const std::int32_t> host_table_rows; // the same rows on the host
    std::int32_t batch = 0;
    std::int32_t width = 0;
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

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
            ForwardState state, ForwardKV kv, ForwardExperts experts, std::int32_t max_context);

    // Logits of each sequence's last column: BF16 [V, batch].
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
};

} // namespace ninfer::models::qwen4_exp::execution
