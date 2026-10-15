#pragma once

#include "models/qwen3_5/model.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace infernix::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline ops::LinearPolicy residual_projection_policy(const LinearParameters& p,
                                                                  bool wide_verification) {
    // Keep the neural path's residual activation precision in wide copy verification. A
    // column-invariant projection already runs 16-bit activations at every width.
    return wide_verification && p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                   !ops::column_invariant(p.policy)
               ? ops::LinearPolicy::A16Only
               : p.policy;
}

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    // Prepared once for the Text backbone and its MTP companion; absent for GDN-only data.
    std::optional<ops::PreparedRope> rope;
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    // Set only when the four attention projections share one contiguous parent region,
    // which the fused attn_input_proj / packed-linear path requires. Mixed-format
    // artifacts (e.g. Q8 K/V with bf16 Q/gate) leave this empty.
    std::optional<LinearParameters> packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    ops::PreparedRope rope;
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// The exact route's request invariance runs decode and verification under
// LinearPolicy::A16ColumnInvariant, which needs a column-invariant route for every Text
// projection: single-parent NVFP4 or FP8 attention and GDN input projections, NVFP4 or FP8 output
// and dense MLP projections, an FP8 or Q8 vocabulary head, and the 27B's BF16 GDN control parent.
// Other artifacts run the exact route without request invariance.
[[nodiscard]] bool column_invariant_text_supported(const TextParameters& text);
// `text` with every projection's policy LinearPolicy::A16ColumnInvariant, the GDN control
// included (its fused norm and gating keep one schedule at every width).
[[nodiscard]] TextParameters column_invariant_text(const TextParameters& text);
// The activation precision each projection class may take in batched decode and verification.
// The one-term permission takes precedence over its residual profile.
struct DecodePrecision {
    bool a8          = true;
    bool a4          = true;
    bool a8_residual = false;
    bool a4_residual = false;
};

// `text` with each FP8 projection's policy LinearPolicy::A16Only unless `a8`, and each NVFP4
// projection's unless `a4`; with `a8_residual` (and not `a8`) an FP8 projection permitted 8-bit
// activations takes LinearPolicy::AllowA8Residual, and with `a4_residual` (and not `a4`) an NVFP4
// MLP projection permitted 4-bit activations takes LinearPolicy::AllowA4Residual (others and the
// GDN control keep their own).
[[nodiscard]] TextParameters precision_decode_text(const TextParameters& text,
                                                   DecodePrecision precision);

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    // `text` under LinearPolicy::A16ColumnInvariant, which the exact route's decode and
    // verification run when the artifact supports it (column_invariant_text_supported).
    std::optional<TextParameters> invariant_text;
    // `text` for batched decode and verification under each DecodePrecision, indexed by
    // precision_index (built with invariant_text; precision_decode_text).
    std::array<std::optional<TextParameters>, 9> precision_text;

    [[nodiscard]] static constexpr std::size_t precision_index(DecodePrecision precision) noexcept {
        const std::size_t fp8   = precision.a8 ? 1U : (precision.a8_residual ? 2U : 0U);
        const std::size_t nvfp4 = precision.a4 ? 1U : (precision.a4_residual ? 2U : 0U);
        return fp8 + 3U * nvfp4;
    }

    // The parameters decode and verification run with: the column-invariant copy, else the copy
    // with the selected low-precision permissions.
    [[nodiscard]] const TextParameters& decode_text(bool column_invariant,
                                                    DecodePrecision precision) const noexcept {
        if (column_invariant && invariant_text) { return *invariant_text; }
        const std::size_t index = precision_index(precision);
        if (precision_text[index]) { return *precision_text[index]; }
        return text;
    }
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace infernix::models::qwen3_5::execution
