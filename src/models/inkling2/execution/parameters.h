#pragma once

// The native operands of a loaded Inkling 2 model (AGENTS.md: native Parameters supply the actual operands to
// planning and Program execution): Weights for ops::linear, raw device pointers for the visit-chain, attention and
// expert Ops. Built once from the frozen Model; borrows its device memory.

#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/inkling2_attention.h"
#include "infernix/ops/inkling2_visit.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/w4a4_sparse_moe.h"
#include "models/ablate/parameters.h"
#include "models/inkling2/execution/lora_parameters.h"
#include "models/inkling2/model.h"

#include <array>
#include <optional>
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

// The DSpark drafter's operands (bound with --spec dspark).
struct DSparkLayerOperands {
    bool sliding = false;
    const std::uint16_t *context_norm = nullptr, *input_norm = nullptr, *post_attention_norm = nullptr; // BF16 [H]
    Weight query, key, value;           // the block's projections (one parent)
    Weight context_key, context_value;  // the context features' projections (one parent)
    const std::uint16_t *query_norm = nullptr, *key_norm = nullptr, *context_key_norm = nullptr; // BF16 [d]
    Weight output;                      // [H, heads * d]
    Weight gate, up, down;              // SwiGLU [I, H], [I, H], [H, I]
};

struct DSparkOperands {
    const std::uint16_t* feature_weights = nullptr; // BF16 [layers + 1][features] (softmax per row)
    const std::uint16_t* mask_embedding  = nullptr; // BF16 [H]
    const std::uint16_t* final_norm      = nullptr; // BF16 [H]
    std::vector<DSparkLayerOperands> layers;
    const std::uint16_t* markov_context_norm    = nullptr; // BF16 [H]
    const std::uint16_t* markov_embedding       = nullptr; // BF16 [V][rank]
    Weight markov_projection;                              // [3 rank, 2 rank + 2 H]
    const std::uint16_t* markov_projection_bias = nullptr; // BF16 [3 rank]
    Weight fold;                                           // [V, rank]
    const std::uint16_t* fold_bias              = nullptr; // BF16 [V]
};

// The DFlash2 drafter's operands (bound with --spec dflash2 --draft-model), in the fused DFlash2 Ops' types
// (execution/dflash2.cpp): Q8 projections but the convolutions' kernel projections and the selector's (BF16).
struct DFlash2LayerOperands {
    Tensor input_norm, post_attention_norm;   // BF16 [H]
    Weight query_key_value;                   // Q8 [q + kv + kv, H] (ops::attn_input_proj's parent)
    Weight context_key, context_value;        // Q8 [kv, H]: the K/V words, shared
    Tensor query_norm, key_norm;              // BF16 [d]
    Weight output;                            // Q8 [H, q]
    Weight gate_up, down;                     // Q8 [2I, H] (gate rows, then up), [H, I]
    Tensor attention_base, mlp_base;          // BF16 [H, taps, sides]
    Weight attention_projection, mlp_projection; // BF16 [2 taps H / group, H]
};

struct DFlash2Operands {
    ops::PreparedRope rope;    // full-head 1-D, the drafter's theta
    Weight feature_projection; // Q8 [H, features * H]
    Tensor context_norm, final_norm;
    std::vector<DFlash2LayerOperands> layers;
    Weight selector_hidden;                          // BF16 [rank, H]
    Tensor predecessor_codebook, successor_codebook; // BF16 [rank, V]
};

// The vision tower's operands (bound with --vision): BF16 vectors [H] unless noted.
struct VisionLayerOperands {
    const std::uint16_t *attention_norm = nullptr, *attention_norm_bias = nullptr;
    Weight qkv;                              // [3H, H]: q, k, v rows (one parent)
    const std::uint16_t* qkv_bias = nullptr; // [3H]
    Weight output;                           // [H, H]
    const std::uint16_t *output_bias = nullptr, *attention_scale = nullptr;
    const std::uint16_t *mlp_norm = nullptr, *mlp_norm_bias = nullptr;
    Weight up;                               // [I, H]
    const std::uint16_t* up_bias = nullptr;  // [I]
    Weight down;                             // [H, I]
    const std::uint16_t *down_bias = nullptr, *mlp_scale = nullptr;
};
struct VisionOperands {
    Weight patch_embedding;                        // [H, 3 p^2]
    const std::uint16_t* position_table = nullptr; // [grid^2][H]
    const std::uint16_t* class_row      = nullptr; // [H]
    const std::uint16_t *pre_norm = nullptr, *pre_norm_bias = nullptr;
    std::vector<VisionLayerOperands> layers;
    Weight projector_in, projector_out;            // [P, m^2 H], [text H, P]
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
    std::optional<DSparkOperands> dspark;   // --spec dspark
    std::optional<DFlash2Operands> dflash2; // --spec dflash2
    std::optional<VisionOperands> vision;   // --vision
    std::optional<LoraParameters> lora;     // --lora (docs/lora.md): the adapter's device factors
    std::optional<AblateParameters> ablate; // --ablate (docs/ablate.md): the direction's device copy, each layer's cuts

    // A lane's cut at `layer` (a physical layer, the same on every pass): inactive without --ablate or where the sidecar
    // leaves the lane uncut.
    [[nodiscard]] ops::inkling2::ResidualCut ablate_cut(std::uint32_t layer, AblateParameters::Lane lane) const noexcept {
        if (!ablate) { return {}; }
        const ops::DirectionCut cut = ablate->cut(layer, lane);
        return {cut.direction, cut.alpha};
    }
    // The device bytes of the adapter factors and the sidecar's direction (MemorySummary::adapter_bytes).
    [[nodiscard]] std::uint64_t adapter_bytes() const noexcept {
        return (lora ? lora->factors.bytes : 0) + (ablate ? ablate->direction.bytes : 0);
    }
};

} // namespace infernix::models::inkling2
