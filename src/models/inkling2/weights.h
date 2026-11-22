#pragma once

// Logical parameter handles of a loaded Inkling 2 model. Names follow the converter (tools/convert/inkling2_loop.py,
// design docs/maintainer/inkling2_mini-design.md §18); handles index the frozen Model's bound weights.

#include "core/weight_view.h"
#include "infernix/ops/linear.h"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace infernix::models::inkling2 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
    [[nodiscard]] bool valid() const noexcept { return index != std::numeric_limits<std::size_t>::max(); }
};

struct BoundWeight {
    std::string name;
    WeightView view;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only; // of the first Use, if any
};

// One layer's attention. query, relative, gate, key and value share the K2 parent [q + r + q + k + v, H] (rows bound
// to x_q, x_q, x_q, x_k, x_v); sink (and on full layers key_taps and value_taps) a small parent beside it.
struct AttentionWeights {
    WeightId query, relative, gate, key, value, sink;
    WeightId key_taps, value_taps;                 // full layers: the dynamic convs' per-token taps [kv * k, H]
    WeightId output;                               // [H, q]
    WeightId query_scale, relative_scale;          // BF16 [heads, head_dim], [heads, relative_dim]
    WeightId relative_table;                       // BF16 [relative_dim, relative_extent]
    WeightId key_conv, value_conv;                 // BF16 [k, kv * head_dim], lag-major (row 0 the current token)
};

// A loop message gate (core layers): z = down [rms(x); rms(m)], x' = (1 - tanh(forget z)) x + tanh(gain z) m.
struct MessageGateWeights {
    WeightId down;          // [rank, 2H]; the four kinds' downs share one parent
    WeightId forget, gain;  // [H, rank], one parent per kind
};

struct MessageWeights {
    WeightId queries;                        // FP32 [4, H]: (query, key, value, mlp) cross-pass mix queries
    std::array<MessageGateWeights, 4> gates; // query, key, value, mlp
};

struct MoeWeights {
    WeightId router;          // BF16 [E, H]
    WeightId router_bias;     // FP32 [E]
    WeightId experts;         // nvfp4_mul in nvfp4_expert_rg16_v1, or fp8_e4m3fn_g32_e8m0 in mxfp8_expert_rg16_v1
    WeightId expert_amax;     // NVFP4 banks: FP32 [E, 2] (gate/up, down), the checkpoint's words
    WeightId expert_denominators; // NVFP4 banks: FP32 [2]
};

struct LayerWeights {
    WeightId attention_mix_query; // FP32 [H] (sliding) or [3, H] (full: q, k, v); invalid at layer 0
    WeightId mlp_mix_query;       // FP32 [H]
    WeightId attention_norm;      // sliding layers
    WeightId query_norm, key_norm, value_norm; // full layers
    WeightId mlp_norm;
    AttentionWeights attention;
    WeightId attention_conv, mlp_conv; // BF16 [k, H], lag-major
    std::optional<MessageWeights> messages; // core layers
    MoeWeights moe;
};

// One DSpark drafter layer (design §13): the block's q, k, v (one parent, bound to its hidden input), the context's
// k and v (one parent, bound to the layer's mixed context features), the per-head norms, o, and the SwiGLU MLP.
struct DSparkLayerWeights {
    WeightId context_norm, input_norm, post_attention_norm; // BF16 [H]
    WeightId query, key, value;                             // [heads * d, H], [kv * d, H], [kv * d, H]
    WeightId context_key, context_value;                    // [kv * d, H]
    WeightId query_norm, key_norm, context_key_norm;        // BF16 [d]
    WeightId output;                                        // [H, heads * d]
    WeightId gate, up, down;                                // [I, H], [I, H], [H, I]
};

// The DSpark drafter: the context features' mixing weights (softmax per row: a row per layer, then the Markov head's),
// the block's mask embedding, the final norm, the layers and the Markov RNN head (its token embedding, the joint
// projection [3 rank, 2 rank + 2 H] and bias, the fold of the target head through its output projection [V, rank]
// and its bias [V]).
struct DSparkWeights {
    WeightId feature_weights; // [layers + 1, features]
    WeightId mask_embedding;  // BF16 [H]
    WeightId final_norm;      // BF16 [H]
    std::vector<DSparkLayerWeights> layers;
    WeightId markov_context_norm;     // BF16 [H]
    WeightId markov_embedding;        // BF16 [V, rank]
    WeightId markov_projection;       // [3 rank, 2 rank + 2 H]
    WeightId markov_projection_bias;  // [3 rank]
    WeightId fold;                    // [V, rank]
    WeightId fold_bias;               // [V]
};

// One vision layer: LayerNorm (with bias), q/k/v [H, H] each with biases (q, k, v one parent, their biases another:
// one GEMM, one bias pass), the output projection and bias, the attention branch's layer scale; LayerNorm, the MLP
// (up [I, H] + bias, exact GELU, down [H, I] + bias) and its layer scale. Vectors BF16 [H] unless noted.
struct VisionLayerWeights {
    WeightId attention_norm, attention_norm_bias;
    WeightId query, key, value;
    WeightId query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId attention_scale;
    WeightId mlp_norm, mlp_norm_bias;
    WeightId up, up_bias; // [I, H], [I]
    WeightId down, down_bias;
    WeightId mlp_scale;
};
// The vision tower (design §15): the patch embedding [H, 3 p^2] (columns in (ky, kx, channel) order), the learned
// position table BF16 [grid^2, H] (row gy * grid + gx), the class token's row (class + its position, BF16 [H]), the
// pre-norm, the layers, and the projector (in [P, m^2 H], exact GELU, out [text H, P]; no biases).
struct VisionWeights {
    WeightId patch_embedding;
    WeightId position_embedding;
    WeightId class_row;
    WeightId pre_norm, pre_norm_bias;
    std::vector<VisionLayerWeights> layers;
    WeightId projector_in, projector_out;
};

struct TextWeights {
    WeightId token_embedding; // BF16 [V, H]
    WeightId embedding_norm;  // BF16 [H]
    WeightId output_head;     // BF16 [V, H]
    WeightId final_norm;      // BF16 [H]
    WeightId output_query;    // FP32 [H]
    std::vector<LayerWeights> layers;
    std::optional<DSparkWeights> dspark; // bound with --spec dspark
    std::optional<VisionWeights> vision; // bound with --vision
};

} // namespace infernix::models::inkling2
