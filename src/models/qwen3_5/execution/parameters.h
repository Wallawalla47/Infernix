#pragma once

#include "core/arena.h"
#include "models/qwen3_5/model.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace infernix::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline ops::LinearPolicy residual_projection_policy(const LinearParameters& p,
                                                                  bool wide_verification) {
    // Keep the neural path's residual activation precision in wide copy verification. A
    // column-invariant projection already runs 16-bit activations at every width.
    return wide_verification && is_fp8_row(p.weight.qtype) &&
                   !ops::column_invariant(p.policy)
               ? ops::LinearPolicy::A16Only
               : p.policy;
}

// The compute policy a projection's single parent or pair runs with.
[[nodiscard]] inline ops::LinearPolicy projection_policy(const ops::ProjectionWeights& projection) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&projection)) {
        return pair->policy;
    }
    return std::get<LinearParameters>(projection).policy;
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

// An offloaded MoE layer (weights.h OffloadedMoeWeights; ops/offloaded_sparse_moe.h): the router
// rows and the shared-expert gate row as BF16 [H, E] and [H, 1] tensors (projection_fp32's form),
// the Model's routed bank (pinned host memory) and its shared expert's bank (device memory), and the
// layer's index in the expert cache. The Program supplies each call's mutable expert source (VRAM
// frames, staging slots, CPU channel). Every expert route is exact W4A16 at every width, so the
// layer has no compute policy and is column-invariant as it is.
struct OffloadedMoeParameters {
    Tensor router, shared_score;
    const runtime::expert_offload::ExpertBank* bank        = nullptr;
    const runtime::expert_offload::ExpertBank* shared_bank = nullptr;
    // Device [experts + 1]: the routed experts' multipliers, then the shared expert's (the call's
    // last job, MoeExpertSource::scales). Every route reads these (the lossless one too: W4A16
    // banks store no input scales).
    const ops::offloaded_moe::ExpertScales* device_scales = nullptr;
    std::uint32_t layer   = 0;
    std::uint32_t experts = 0;
    std::uint32_t top_k   = 0;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights, OffloadedMoeParameters>;

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
// LinearPolicy::A16ColumnInvariant. A projection whose problem registers no dedicated
// column-invariant route runs its exact one-column route once per column (slower, same guarantee).
// One entry per role, format and shape, with the number of projections it covers.
struct ColumnByColumnProjection {
    std::string role;
    QType qtype = QType::BF16;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int32_t count = 0;
};
[[nodiscard]] std::vector<ColumnByColumnProjection>
column_by_column_projections(const TextParameters& invariant_text,
                             const MtpParameters* invariant_mtp = nullptr,
                             const LinearParameters* invariant_proposal_head = nullptr);
// `text` with every projection's policy LinearPolicy::A16ColumnInvariant: paired parents, the
// sparse MoE and the GDN control included (its fused norm and gating keep one schedule at every
// width).
[[nodiscard]] TextParameters column_invariant_text(const TextParameters& text);
// `mtp` likewise: the drafter of batched decode and verification rounds, whose draft tokens decide
// which sampled tokens a round accepts.
[[nodiscard]] MtpParameters column_invariant_mtp(const MtpParameters& mtp);
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
    // With offloaded experts: every offloaded layer's device scales, [layers][experts + 1]
    // (OffloadedMoeParameters::device_scales points into it).
    DeviceBuffer expert_scales;
    TextParameters text;
    // `text` under LinearPolicy::A16ColumnInvariant, which the exact route's decode and
    // verification run (every artifact), and its projections that run column by column.
    TextParameters invariant_text;
    std::vector<ColumnByColumnProjection> column_by_column;
    // `text` for batched decode and verification under each DecodePrecision, indexed by
    // precision_index (precision_decode_text).
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
        if (column_invariant) { return invariant_text; }
        const std::size_t index = precision_index(precision);
        if (precision_text[index]) { return *precision_text[index]; }
        return text;
    }
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
    // `mtp` and the proposal head under LinearPolicy::A16ColumnInvariant, which the exact route's
    // batched draft calls run with (TextContext::mtp_forward_decode_batch, mtp_propose_batch).
    std::optional<MtpParameters> invariant_mtp;
    std::optional<LinearParameters> invariant_proposal_head;
};

} // namespace infernix::models::qwen3_5::execution
