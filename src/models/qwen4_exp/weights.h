#pragma once

// Logical parameter handles of a loaded Qwen4Exp model. Names follow the converter
// (tools/convert/qwen4_exp.py); handles index the frozen Model's bound weights.

#include "core/weight_view.h"
#include "infernix/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace infernix::models::qwen4_exp {

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
    // FP32 [experts, 3]: gate, up, down activation global scales of a W4A4 bank (its use allows A4);
    // not bound for a W4A16 bank (its use is A16-only: experts stored without activation scales).
    WeightId input_scales;
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

// The MTP drafter (design §11.2): its input fusion, one QSA block whose routed experts are
// device-resident row ranges of two parents, and its own final mixer; it shares the embedding
// and the head (or the proposal head) with the text model.
struct MtpWeights {
    WeightId embedding_norm, hidden_norm;               // BF16 [H], [S*H]
    WeightId embedding_projection, hidden_projection;   // [H, H]
    HyperConnectionWeights attn_hc, mlp_hc, final_mixer;
    AttentionWeights attention;
    WeightId router, shared_score, shared_gate, shared_up, shared_down;
    std::vector<WeightId> expert_gate_up; // per expert: gate, then up ([I, H] each)
    std::vector<WeightId> expert_down;    // per expert: [H, I]
};

// One DFlash2 layer (docs/maintainer/dflash.md): its dynamic convolutions' base kernels BF16
// [2 sides, taps, H] and coefficient projections, the query block's Q/K/V parent and the context
// K/V (the converter shares the K/V words between the two roles), and the gate/up parent.
struct DFlash2LayerWeights {
    WeightId input_norm, post_attention_norm;
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
    WeightId gate, up, down;
    WeightId attention_base, attention_projection, mlp_base, mlp_projection;
};

// The DFlash2 drafter of a companion artifact (--draft-model); it shares the token embedding and
// the head (or the proposal head) with the text model.
struct DFlash2Weights {
    WeightId feature_projection; // [H, taps * H]
    WeightId context_norm, final_norm;
    std::vector<DFlash2LayerWeights> layers;
    WeightId selector_hidden;                            // [rank, H]
    WeightId predecessor_codebook, successor_codebook;   // BF16 [vocab, rank]
};

// The optimized proposal head (--lm-head-draft): lm_head rows of a frequency shortlist.
struct ProposalWeights {
    WeightId head;      // [rows, H]
    WeightId token_ids; // I32 [rows]: each row's token id
    // The converter's 2-bit copy of its rows (screen/proposal_head, q2_g64_fp16 [rows, H]), bound for the
    // MTP drafter when present (LoadOptions::proposal_head_screen); invalid otherwise.
    WeightId screen;
};

// The Vision tower (design §19.3.2): Qwen3.5's tower and parameter names, BF16, pinned host memory
// with vision offload, device memory otherwise.
struct VisionNormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    VisionNormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    VisionNormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

struct TextWeights {
    WeightId token_embedding; // BF16, pinned host memory
    WeightId output_head;
    // The converter's 2-bit copy of output_head (screen/output_head, q2_g64_fp16 [V, H]), bound for the
    // screened-lm-head fast path when present and the head is one it rescores (LoadOptions::lm_head_screen,
    // screened_lm_head); invalid otherwise.
    WeightId output_head_screen;
    HyperConnectionWeights final_mixer;
    std::vector<BlockWeights> layers;
    std::optional<MtpWeights> mtp;
    std::optional<DFlash2Weights> dflash2; // bound from the companion artifact
    std::optional<ProposalWeights> proposal;
    std::optional<VisionWeights> vision;
};

} // namespace infernix::models::qwen4_exp
