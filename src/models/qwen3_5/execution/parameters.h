#pragma once

#include "core/arena.h"
#include "models/ablate/parameters.h"
#include "models/qwen3_5/model.h"
#include "infernix/ops/lora.h"
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
    // column-invariant projection already runs 16-bit activations at every width; an A8Only one
    // (`--a8-activations`) keeps its 8-bit activations at every width.
    return wide_verification && is_fp8_row(p.weight.qtype) &&
                   !ops::column_invariant(p.policy) && !ops::a8_only(p.policy)
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

// A LoRA adapter's operands at one projection input (models/qwen3_5/load/lora.h): A stacked over the adapters that read
// the input (ops::lora_shrink) and the adapted parent's term. The call binds the term's U (addend.u, the shrink's
// output over the call's columns); addend.u_rows is a.rows.
using LoraSiteParameters = ops::LoraSite;

// --lora on the --lossless route (docs/maintainer/lossless-route.md §2, LoRA adapter): a site's A as its three exact
// BF16 planes (ops::lossless_lora_split3: [a_rows, 3 k]) and the adapted parent's FP32 B ([parent rows, rank], the
// term's row order), with the term's segments (row ends, U rows, scales) as ops::LoraAddend gives them.
struct LosslessLoraSite {
    const void* a3      = nullptr;
    std::int32_t a_rows = 0;
    std::int32_t k      = 0;
    const float* b      = nullptr;
    std::int32_t rank   = 0;
    std::int32_t segments = 0;
    std::array<ops::LoraSegment, 4> segment{};
};

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
    // --lora (text layers only): gate_proj/up_proj over the gate/up parent, down_proj over down; on the lossless
    // route the lossless_ forms instead.
    std::optional<LoraSiteParameters> lora_gate_up, lora_down;
    std::optional<LosslessLoraSite> lossless_lora_gate_up, lossless_lora_down;
};

// An offloaded MoE layer (weights.h OffloadedMoeWeights; ops/offloaded_sparse_moe.h): the router
// rows and the shared-expert gate row as BF16 [H, E] and [H, 1] tensors (projection_fp32's form),
// the Model's routed bank (pinned host memory) and its shared expert's bank (device memory) or dense
// projections, and the layer's index in the expert cache. The Program supplies each call's mutable
// expert source (VRAM frames, staging slots, CPU channel). Every expert route is exact W4A16 at every
// width, so the routed experts have no compute policy and are column-invariant as they are; dense
// shared projections carry their own policies (column-invariant on the exact route).
struct OffloadedMoeParameters {
    Tensor router, shared_score;
    const runtime::expert_offload::ExpertBank* bank        = nullptr;
    const runtime::expert_offload::ExpertBank* shared_bank = nullptr; // or
    std::optional<DenseParameters> shared;                            // (the dispatch then has no shared job)
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
    // --lora: q_proj/k_proj/v_proj over the projection (addend.b in the single parent's row order query, key, gate,
    // value; a paired projection's second parent starts at gate), o_proj over output; on the lossless route the
    // lossless_ forms instead.
    std::optional<LoraSiteParameters> lora_input, lora_output;
    std::optional<LosslessLoraSite> lossless_lora_input, lossless_lora_output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
    // --lora: one shrink of h stacked over in_proj_qkv, in_proj_z, in_proj_a and in_proj_b (lora_input.a); the
    // projection's term (lora_input.addend, B in the single parent's q/k/value/z row order; absent when neither
    // in_proj_qkv nor in_proj_z is adapted) and the control's (lora_control: a rows, then b rows) over its U;
    // out_proj over the output. On the lossless route the lossless_ forms instead (the input site's segments: the
    // q/k/value channels, z, a, b).
    std::optional<LoraSiteParameters> lora_input, lora_output;
    ops::LoraAddend lora_control;
    std::optional<LosslessLoraSite> lossless_lora_input, lossless_lora_output;
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

// A DSpine drafter's code bridge (docs/maintainer/dspine.md): each layer's read [C, H] and write
// [H, R], the projection [3R, C], the gate's bias BF16 [R], the last write's columns BF16 [H, R]
// (column r is write[:, r]) and every token's sender rows BF16 [2R, V].
struct DSpineBridgeLayerParameters {
    LinearParameters read, write;
};

struct DSpineBridgeParameters {
    LinearParameters projection;
    Tensor gate_bias, head_write_columns, sender_table;
    std::vector<DSpineBridgeLayerParameters> layers;
};

struct DraftParameters {
    ops::PreparedRope rope;
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    std::optional<DSpineBridgeParameters> dspine;
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
// The one-term permission takes precedence over its residual profile; `a8_subset` grants the
// one-term 8-bit permission to the attention and GDN input projections and the dense MLP gate/up only.
struct DecodePrecision {
    bool a8          = true;
    bool a4          = true;
    bool a8_residual = false;
    bool a4_residual = false;
    bool a8_subset   = false;
};

// `text` with each FP8 projection's policy LinearPolicy::A16Only unless `a8` (or `a8_subset` and
// the projection is an input projection or an MLP gate/up), and each NVFP4 projection's unless `a4`;
// with `a8_residual` an FP8 projection permitted 8-bit activations and not given one term takes
// LinearPolicy::AllowA8Residual, and with `a4_residual` (and not `a4`) an NVFP4 MLP projection
// permitted 4-bit activations takes LinearPolicy::AllowA4Residual (others and the GDN control keep
// their own).
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
    std::array<std::optional<TextParameters>, 15> precision_text;

    // The FP8 permission: none, one term, the residual profile, or the one-term subset over the
    // residual profile or over 16 bits (5 forms), by the NVFP4 permission's 3.
    [[nodiscard]] static constexpr std::size_t precision_index(DecodePrecision precision) noexcept {
        const std::size_t fp8   = precision.a8          ? 1U
                                  : precision.a8_subset ? (precision.a8_residual ? 3U : 4U)
                                                        : (precision.a8_residual ? 2U : 0U);
        const std::size_t nvfp4 = precision.a4 ? 1U : (precision.a4_residual ? 2U : 0U);
        return fp8 + 5U * nvfp4;
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
    // The DFlash2 drafter's full proposal head screened (candidates RV-41): its 2-bit copy, fitted at load
    // (ops::screened_head_fit) where the drafter's top sixteen run over the full head (not on the lossless route) and
    // the head is a 5,120-wide FP8 row or q8 head: ops::screened_head_topk serves its calls of at most 16 columns.
    DeviceBuffer draft_screen;
    std::optional<Weight> draft_head_screen;
    // --lora: every adapted layer's factors in their device layout (prepare_lora: BF16, or on the lossless route A's
    // BF16 planes and FP32 B); the text layers' sites point into it. A plan-only model allocates nothing and counts
    // the bytes in planned_lora_bytes.
    DeviceBuffer lora_factors;
    // --spec dspine: the head-write table BF16 [R, V] (dspine_head_write), column v of which is the target
    // head's row v through the drafter's last write scaled by its final norm's weight; derived once here
    // from the target's head (243 MiB for the Qwen3.8-27B drafter). Empty otherwise.
    DeviceBuffer dspine_head_write_storage;
    Tensor dspine_head_write;
    // A plan-only model's (Model::weights_resident false) device bytes these parameters would allocate at a start
    // and did not: the draft screen's and the adapter's. Engine::plan counts them beside the weights.
    std::uint64_t planned_draft_screen_bytes = 0;
    std::uint64_t planned_lora_bytes         = 0;
    // The device bytes beside the weights' arena, allocated (a start) or planned (a plan-only model): the one source
    // of these figures for sizing, the plan, the memory summary and the memory parts.
    [[nodiscard]] std::uint64_t draft_screen_bytes() const noexcept {
        return draft_screen.bytes + planned_draft_screen_bytes;
    }
    [[nodiscard]] std::uint64_t lora_bytes() const noexcept { return lora_factors.bytes + planned_lora_bytes; }
    // --spec dspine: the head-write table, allocated or (a plan-only model) planned.
    std::uint64_t planned_dspine_head_write_bytes = 0;
    [[nodiscard]] std::uint64_t dspine_head_write_bytes() const noexcept {
        return dspine_head_write_storage.bytes + planned_dspine_head_write_bytes;
    }
    // --ablate (docs/ablate.md): the sidecar's direction on the device and each layer's two cuts; nullopt without one
    // (and on a plan-only model, which counts the direction's bytes in planned_ablate_bytes).
    std::optional<models::AblateParameters> ablate;
    std::uint64_t planned_ablate_bytes = 0;
    [[nodiscard]] std::uint64_t ablate_bytes() const noexcept {
        return (ablate ? ablate->direction.bytes : 0) + planned_ablate_bytes;
    }
    // The adapters' device bytes beside the weights' arena (MemorySummary::adapter_bytes): a LoRA adapter's factors and
    // an ablation sidecar's direction, allocated or planned; allocated_adapter_bytes counts the allocated ones only.
    [[nodiscard]] std::uint64_t adapter_bytes() const noexcept { return lora_bytes() + ablate_bytes(); }
    [[nodiscard]] std::uint64_t allocated_adapter_bytes() const noexcept {
        return lora_factors.bytes + (ablate ? ablate->direction.bytes : 0);
    }
    // A layer's lane cut: inactive without --ablate or where the sidecar leaves the lane uncut.
    [[nodiscard]] ops::DirectionCut ablate_cut(std::size_t layer, models::AblateParameters::Lane lane) const noexcept {
        return ablate ? ablate->cut(static_cast<std::uint32_t>(layer), lane) : ops::DirectionCut{};
    }
    std::optional<ProposalParameters> proposal;
    // `mtp` and the proposal head under LinearPolicy::A16ColumnInvariant, which the exact route's
    // batched draft calls run with (TextContext::mtp_forward_decode_batch, mtp_propose_batch).
    std::optional<MtpParameters> invariant_mtp;
    std::optional<LinearParameters> invariant_proposal_head;
};

} // namespace infernix::models::qwen3_5::execution
