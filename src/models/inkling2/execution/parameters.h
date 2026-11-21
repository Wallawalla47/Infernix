#pragma once

// The native operands of a loaded Inkling 2 model (AGENTS.md: native Parameters supply the actual operands to
// planning and Program execution): Weights for ops::linear, raw device pointers for the visit-chain, attention and
// expert Ops. Built once from the frozen Model; borrows its device memory.

#include "core/weight.h"
#include "infernix/ops/inkling2_attention.h"
#include "infernix/ops/w4a4_sparse_moe.h"
#include "models/inkling2/model.h"

#include <array>
#include <cstdint>
#include <vector>

namespace infernix::models::inkling2 {

struct AttentionOperands {
    Weight query, relative, gate, key, value, sink; // K2: rows bound to x_q (query, relative, gate, sink), x_k, x_v
    Weight key_taps, value_taps;                    // full layers: the dynamic convs' per-token taps
    Weight output;                                  // W_o [H, 3072]
    ops::inkling2::AttentionLayer layer;            // scales, relative table, K/V conv taps, the kind
};

// A loop message gate: z = down [rms(x); rms(m)] ([rank, 2H]) and the forget and gain projections of z as one
// [2H, rank] matrix (forget rows, then gain rows: the converter's parent order).
struct GateOperands {
    Weight down;
    Weight forget_gain;
};

struct LayerOperands {
    bool full     = false;
    bool boundary = false; // the layer opens an attention-residual block
    ops::w4a4_moe::ExpertFormat format = ops::w4a4_moe::ExpertFormat::Nvfp4FourOverSix;
    const float* attention_mix_query = nullptr; // FP32 [nq][H]; null at layer 0
    const std::uint16_t* norms[3]    = {};      // sliding: the attention norm three times; full: q, k, v
    const std::uint16_t* mlp_norm    = nullptr;
    const float* mlp_mix_query       = nullptr;
    const std::uint16_t* attention_conv = nullptr; // [4][H]
    const std::uint16_t* mlp_conv       = nullptr; // [4][H]
    AttentionOperands attention;
    bool messages                 = false;      // a core layer (its loop messages run on passes 2 and later)
    const float* message_queries  = nullptr;    // FP32 [4][H]: q, k, v, mlp
    std::array<GateOperands, 4> gates;          // q, k, v, mlp
    const std::uint16_t* router   = nullptr;    // BF16 [E][H]
    const float* router_bias      = nullptr;    // FP32 [E]
    ops::w4a4_moe::Bank bank;
};

struct Parameters {
    explicit Parameters(const Model& model);

    const Model& model;
    const std::uint16_t* embedding      = nullptr; // BF16 [V][H]
    const std::uint16_t* embedding_norm = nullptr;
    const std::uint16_t* final_norm     = nullptr;
    const float* output_query           = nullptr;
    Weight head;                                   // BF16 [V][H], FP32 logits
    std::vector<LayerOperands> layers;
};

} // namespace infernix::models::inkling2
