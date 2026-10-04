#pragma once

// Logical parameter handles of a loaded Qwen4Exp model. Names follow the converter
// (tools/convert/qwen4_exp.py); handles index the frozen Model's bound weights.

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp {

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

// One hyper-connection mixer: grouped norm, low-rank input mix, and (except the final mixer) the
// per-stream injection weights. `down` and `inject` share one parent ([rank + streams, width]).
struct HyperConnectionWeights {
    WeightId norm, down, up;
    WeightId inject; // invalid for the final mixer
};

// QSA block: query/gate/key/value and the indexer's query/key share one input parent.
struct AttentionWeights {
    WeightId query, gate, key, value, index_query, index_key;
    WeightId query_norm, key_norm, index_query_norm, index_key_norm, output;
};

struct GdnWeights {
    WeightId query, key, value, z;
    WeightId a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

struct MoeWeights {
    WeightId router, shared_score;
    WeightId shared_gate, shared_up, shared_down;
    WeightId experts;      // nvfp4_mul bank in nvfp4_expert_rg16_v1, pinned host memory
    WeightId input_scales; // FP32 [experts, 3]: gate, up, down activation global scales
};

struct PleWeights {
    WeightId key_projection, value_projection;
    WeightId key_norm, query_norm, conv_norm, convolution, ngram_scale;
};

struct BlockWeights {
    HyperConnectionWeights attn_hc, mlp_hc;
    std::variant<AttentionWeights, GdnWeights> mixer;
    MoeWeights moe;
    std::optional<PleWeights> ple;
};

struct TextWeights {
    WeightId token_embedding; // BF16, pinned host memory
    WeightId output_head;
    HyperConnectionWeights final_mixer;
    std::vector<BlockWeights> layers;
};

} // namespace ninfer::models::qwen4_exp
