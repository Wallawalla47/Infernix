#include "models/qwen3_5/execution/parameters.h"

#include "core/temporary_switches.h"
#include "core/weight_view.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    // `expert_scales`: the Parameters' device scales of the offloaded layers (null without them).
    explicit Prepare(const Model& model, const ops::offloaded_moe::ExpertScales* expert_scales = nullptr)
        : model_(model), expert_scales_(expert_scales) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    // `layer` is the block's text layer (the expert cache's index of an offloaded layer).
    FfnParameters ffn(const BlockWeights& w, std::size_t layer = 0) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        if (const auto* offloaded = std::get_if<OffloadedMoeWeights>(&w.ffn)) {
            const auto& config = std::get<MoeConfig>(model_.config().text.ffn);
            OffloadedMoeParameters out;
            out.router       = tensor(offloaded->router);
            out.shared_score = tensor(offloaded->shared_score);
            out.bank         = &model_.expert_banks()[layer];
            if (offloaded->shared) {
                out.shared = dense(*offloaded->shared);
            } else {
                out.shared_bank = &model_.shared_expert_banks()[layer];
            }
            out.layer        = static_cast<std::uint32_t>(layer);
            out.experts      = config.num_experts;
            out.top_k        = config.num_experts_per_tok;
            if (expert_scales_ == nullptr) { throw std::logic_error("offloaded MoE: the device scales are missing"); }
            out.device_scales = expert_scales_ + layer * (static_cast<std::size_t>(config.num_experts) + 1);
            const auto h     = static_cast<std::int32_t>(model_.config().text.hidden_size);
            if (out.router.dtype != DType::BF16 || out.router.ne[0] != h ||
                out.router.ne[1] != static_cast<std::int32_t>(config.num_experts) ||
                out.shared_score.dtype != DType::BF16 || out.shared_score.ne[0] != h || out.shared_score.ne[1] != 1) {
                throw std::invalid_argument("offloaded MoE: the router and shared gate must be BF16 rows of H");
            }
            if (config.num_experts > 512 || config.num_experts_per_tok > 16) {
                throw std::invalid_argument("offloaded MoE routes up to 512 experts and top-16");
            }
            return out;
        }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(model_.input(expert.gate));
            gate_up.push_back(model_.input(expert.up));
            down.push_back(model_.input(expert.down));
        }
        // The closed SparseMoe Op takes the 35B-A3B geometry; another (the 122B-A10B's drafter) runs
        // its routed banks through resident_moe_experts (ffn.cpp).
        const auto& text   = model_.config().text;
        const auto& config = std::get<MoeConfig>(text.ffn);
        const bool closed  = text.hidden_size == 2048 && config.num_experts == 256 &&
                            config.moe_intermediate_size == 512 && config.shared_expert_intermediate_size == 512;
        return with_context(model_.weight(moe.router).name, [&] {
            const auto prepare = closed ? &ops::prepare_sparse_moe_weights : &ops::prepare_resident_moe_weights;
            return prepare(model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                           model_.input(moe.shared.gate), model_.input(moe.shared.up),
                           model_.input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w, std::size_t layer = 0) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w, layer);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{
                ops::prepare_attn_input_proj_weights(model_.input(a->query), model_.input(a->key),
                                                     model_.input(a->gate), model_.input(a->value)),
                tensor(a->query_norm), tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{
                ops::prepare_gdn_input_proj_weights(model_.input(g.query), model_.input(g.key),
                                                        model_.input(g.value), model_.input(g.z)),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                         model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    // True when the four attention projections address one contiguous parent region, which
    // the fused packed / attn_input_proj path requires. Mixed-format banks (e.g. Q8 K/V with
    // bf16 Q/gate) are separate parents and must use the per-projection row path.
    static bool contiguous_bank(const std::array<ops::WeightInput, 4>& inputs) {
        const auto& first = inputs.front().weight;
        if (first.parts.empty()) { return false; }
        const auto* parent = first.parts.front().parent;
        std::uint64_t end  = first.parts.front().begin;
        for (const auto& input : inputs) {
            for (const auto& part : input.weight.parts) {
                if (!parent || part.parent != parent || part.begin != end ||
                    part.end <= part.begin) {
                    return false;
                }
                end = part.end;
            }
        }
        return true;
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        if (contiguous_bank(inputs)) {
            out.projection.packed = ops::prepare_linear_weight(inputs);
        }
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        const auto& config = *model_.config().draft;
        out.rope = ops::prepare_rope(dimension(config.attention.head_dim), config.rope_theta,
                                    {config.yarn_factor, config.max_position_embeddings});
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
    const ops::offloaded_moe::ExpertScales* expert_scales_;
};

// Every offloaded layer's scales on the device, [layers][experts + 1]: the routed bank's, then the
// shared expert's (its bank holds one expert; a dense shared expert's entry is unused, zero).
DeviceBuffer offloaded_expert_scales(const Model& model) {
    if (!model.offloaded_experts()) { return {}; }
    const auto banks  = model.expert_banks();
    const auto shared = model.shared_expert_banks();
    const auto& moe   = std::get<MoeConfig>(model.config().text.ffn);
    const std::size_t entries = static_cast<std::size_t>(moe.num_experts) + 1;
    if (!shared.empty() && shared.size() != banks.size()) {
        throw std::logic_error("offloaded MoE: the shared expert banks do not cover every layer");
    }
    std::vector<ops::offloaded_moe::ExpertScales> host(banks.size() * entries);
    for (std::size_t layer = 0; layer < banks.size(); ++layer) {
        if (banks[layer].scales.size() != moe.num_experts || (!shared.empty() && shared[layer].scales.size() != 1)) {
            throw std::logic_error("offloaded MoE: a bank's scales differ from its experts");
        }
        std::copy(banks[layer].scales.begin(), banks[layer].scales.end(), host.begin() + layer * entries);
        if (!shared.empty()) { host[layer * entries + moe.num_experts] = shared[layer].scales.front(); }
    }
    DeviceBuffer out(host.size() * sizeof(ops::offloaded_moe::ExpertScales));
    out.copy_from_host(host.data(), host.size() * sizeof(ops::offloaded_moe::ExpertScales));
    return out;
}

} // namespace

Parameters::Parameters(const Model& source) : model(source), expert_scales(offloaded_expert_scales(source)) {
    const Prepare prepare(model, static_cast<const ops::offloaded_moe::ExpertScales*>(expert_scales.p));
    const auto& w        = model.weights();
    const auto& config   = model.config().text;
    if (config.rope_parameters) {
        const auto& rope = *config.rope_parameters;
        text.rope = ops::prepare_rope(dimension(rope.rotary_dim), rope.rope_theta,
                                     {rope.yarn_factor, config.max_position_embeddings});
    }
    // Device memory, or (an artifact with offloaded experts) pinned host memory gathered zero-copy:
    // the rows at the parent's device address either way.
    text.token_embedding = device_native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i], i); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
    // The exact route decodes and verifies request-invariant on every artifact; a projection with
    // no dedicated column-invariant route runs its exact route once per column.
    invariant_text = column_invariant_text(text);
    if (mtp) { invariant_mtp = column_invariant_mtp(*mtp); }
    if (proposal) {
        invariant_proposal_head         = proposal->head;
        invariant_proposal_head->policy = ops::LinearPolicy::A16ColumnInvariant;
    }
    column_by_column =
        column_by_column_projections(invariant_text, invariant_mtp ? &*invariant_mtp : nullptr,
                                     invariant_proposal_head ? &*invariant_proposal_head : nullptr);
    constexpr std::pair<bool, bool> kModes[] = {{false, false}, {true, false}, {false, true}};
    for (const auto [a8, a8_residual] : kModes) {
        for (const auto [a4, a4_residual] : kModes) {
            for (const bool a8_subset : {false, true}) {
                if (a8 && a8_subset) { continue; } // one term everywhere takes precedence
                const DecodePrecision precision{a8, a4, a8_residual, a4_residual, a8_subset};
                precision_text[precision_index(precision)] = precision_decode_text(text, precision);
            }
        }
    }
}

namespace {

void make_invariant(LinearParameters& p) { p.policy = ops::LinearPolicy::A16ColumnInvariant; }

// Keeps the projection's low-precision permission only where its format's class is permitted; a
// projection permitted its class's one-term activations may take its residual profile instead
// (the NVFP4 one, `a4_residual_here`, is registered for the MLP only).
void limit_precision(LinearParameters& p, DecodePrecision precision, bool a4_residual_here) {
    const bool fp8   = is_fp8_row(p.weight.qtype);
    const bool nvfp4 = p.weight.qtype == QType::NVFP4;
    if ((fp8 && precision.a8) || (nvfp4 && precision.a4)) { return; }
    // A static-scale parent (fp8_e4m3fn_row_f32) encodes with its calibrated scale and has no
    // residual profile.
    if (fp8 && p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && precision.a8_residual &&
        ops::allows_a8(p.policy)) {
        p.policy = ops::LinearPolicy::AllowA8Residual;
    } else if (nvfp4 && a4_residual_here && precision.a4_residual && ops::allows_a4(p.policy)) {
        p.policy = ops::LinearPolicy::AllowA4Residual;
    } else {
        p.policy = ops::LinearPolicy::A16Only;
    }
}

void limit_precision(ops::ProjectionWeights& projection, DecodePrecision precision) {
    if (auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
        limit_precision(*single, precision, false);
    }
}

// `precision` for one projection of `kind` in text layer `layer` (-1 for the head): under
// `a8_subset` the projections temporary_switches::a8_subset() covers take one 8-bit term.
DecodePrecision for_projection(DecodePrecision precision, temporary_switches::A8SubsetClass kind,
                               std::int32_t layer) {
    if (precision.a8_subset && temporary_switches::a8_subset().covers(kind, layer)) {
        precision.a8 = true;
    }
    return precision;
}

void make_invariant(ops::ProjectionWeights& projection) {
    if (auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
        make_invariant(*single);
    } else {
        std::get<ops::PairedProjectionWeights>(projection).policy =
            ops::LinearPolicy::A16ColumnInvariant;
    }
}

void make_invariant(FfnParameters& ffn) {
    if (auto* dense = std::get_if<DenseParameters>(&ffn)) {
        make_invariant(dense->gate_up);
        make_invariant(dense->down);
    } else if (auto* moe = std::get_if<ops::SparseMoeWeights>(&ffn)) {
        moe->policy = ops::LinearPolicy::A16ColumnInvariant;
    } else if (auto* offloaded = std::get_if<OffloadedMoeParameters>(&ffn); offloaded && offloaded->shared) {
        // An offloaded MoE's expert routes are exact at every width; its dense shared expert is not.
        make_invariant(offloaded->shared->gate_up);
        make_invariant(offloaded->shared->down);
    }
}

} // namespace

std::vector<ColumnByColumnProjection>
column_by_column_projections(const TextParameters& text, const MtpParameters* mtp,
                             const LinearParameters* proposal_head) {
    std::vector<ColumnByColumnProjection> out;
    const auto note = [&](const char* role, const Weight& weight) {
        for (ColumnByColumnProjection& entry : out) {
            if (entry.role == role && entry.qtype == weight.qtype && entry.rows == weight.n &&
                entry.cols == weight.k) {
                ++entry.count;
                return;
            }
        }
        out.push_back({role, weight.qtype, weight.n, weight.k, 1});
    };
    const Weight& head = text.output_head.weight;
    if (!ops::linear_column_invariant_route(head.qtype, head.n, head.k)) {
        note("output head", head);
    }
    for (const BlockParameters& block : text.layers) {
        // A sparse MoE, every attention input parent, the Q4/Q5 pairs and the GDN control have
        // dedicated routes.
        if (const auto* dense = std::get_if<DenseParameters>(&block.ffn)) {
            const Weight& gate_up = dense->gate_up.weight;
            // Q6 gate/up runs as linear (execution::ffn).
            const bool gate_up_route =
                gate_up.qtype == QType::Q6_G64_FP16
                    ? ops::linear_column_invariant_route(gate_up.qtype, gate_up.n, gate_up.k)
                    : ops::linear_swiglu_column_invariant_route(gate_up.qtype, gate_up.n, gate_up.k);
            if (!gate_up_route) { note("MLP gate/up", gate_up); }
            const Weight& down = dense->down.weight;
            if (!ops::linear_add_column_invariant_route(down.qtype, down.n, down.k)) {
                note("MLP down", down);
            }
        }
        // A dense shared expert beside offloaded experts runs linear_swiglu and linear (offloaded_moe).
        if (const auto* offloaded = std::get_if<OffloadedMoeParameters>(&block.ffn); offloaded && offloaded->shared) {
            const Weight& gate_up = offloaded->shared->gate_up.weight;
            if (!ops::linear_swiglu_column_invariant_route(gate_up.qtype, gate_up.n, gate_up.k)) {
                note("shared expert gate/up", gate_up);
            }
            const Weight& down = offloaded->shared->down.weight;
            if (!ops::linear_column_invariant_route(down.qtype, down.n, down.k)) { note("shared expert down", down); }
        }
        const auto output = [&](const char* role, const LinearParameters& p) {
            if (!ops::linear_add_column_invariant_route(p.weight.qtype, p.weight.n, p.weight.k)) {
                note(role, p.weight);
            }
        };
        if (const auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            output("attention output", attention->output);
        } else {
            const auto& gdn = std::get<GdnParameters>(block.mixer);
            if (const auto* single = std::get_if<LinearParameters>(&gdn.projection);
                single != nullptr && !ops::gdn_input_proj_column_invariant_route(
                                         single->weight.qtype, single->weight.n, single->weight.k)) {
                note("GDN input", single->weight);
            }
            output("GDN output", gdn.output);
        }
    }
    // The drafter's projections run as linear, its K/V row pair as two linear calls; the packed
    // attention input of a MoE drafter (no row views) and its sparse MoE have dedicated routes.
    const auto linear = [&](const char* role, const LinearParameters& p) {
        if (!ops::linear_column_invariant_route(p.weight.qtype, p.weight.n, p.weight.k)) {
            note(role, p.weight);
        }
    };
    if (mtp != nullptr) {
        linear("MTP input", mtp->input_projection);
        if (mtp->projection.packed && mtp->projection.rows) {
            linear("MTP attention input", *mtp->projection.packed);
        } else if (!mtp->projection.packed && mtp->projection.rows) {
            for (const LinearParameters& row : *mtp->projection.rows) { linear("MTP attention input", row); }
        }
        linear("MTP attention output", mtp->output);
        if (const auto* dense = std::get_if<DenseParameters>(&mtp->ffn)) {
            linear("MTP gate/up", dense->gate_up);
            linear("MTP down", dense->down);
        }
        if (proposal_head == nullptr) { linear("MTP head", mtp->output_head); }
    }
    if (proposal_head != nullptr) { linear("MTP proposal head", *proposal_head); }
    return out;
}

TextParameters column_invariant_text(const TextParameters& text) {
    TextParameters out = text;
    make_invariant(out.output_head);
    for (BlockParameters& block : out.layers) {
        make_invariant(block.ffn);
        if (auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            make_invariant(attention->projection);
            make_invariant(attention->output);
        } else {
            auto& gdn = std::get<GdnParameters>(block.mixer);
            make_invariant(gdn.projection);
            make_invariant(gdn.output);
            make_invariant(gdn.control);
        }
    }
    return out;
}

MtpParameters column_invariant_mtp(const MtpParameters& mtp) {
    MtpParameters out = mtp;
    make_invariant(out.input_projection);
    if (out.projection.packed) { make_invariant(*out.projection.packed); }
    if (out.projection.rows) {
        for (LinearParameters& row : *out.projection.rows) { make_invariant(row); }
    }
    make_invariant(out.output);
    make_invariant(out.ffn);
    make_invariant(out.output_head);
    return out;
}

TextParameters precision_decode_text(const TextParameters& text, DecodePrecision precision) {
    using temporary_switches::A8SubsetClass;
    TextParameters out = text;
    // The vocabulary head keeps 16-bit activations unless its class's one-term profile is on.
    const DecodePrecision head = for_projection(precision, A8SubsetClass::Head, -1);
    limit_precision(out.output_head, {head.a8, head.a4, false, false}, false);
    for (std::size_t index = 0; index < out.layers.size(); ++index) {
        BlockParameters& block       = out.layers[index];
        const auto layer             = static_cast<std::int32_t>(index);
        const DecodePrecision input  = for_projection(precision, A8SubsetClass::Input, layer);
        const DecodePrecision output = for_projection(precision, A8SubsetClass::Output, layer);
        // A sparse MoE runs A16 at every policy.
        if (auto* dense = std::get_if<DenseParameters>(&block.ffn)) {
            limit_precision(dense->gate_up, for_projection(precision, A8SubsetClass::GateUp, layer),
                            true);
            limit_precision(dense->down, for_projection(precision, A8SubsetClass::Down, layer), true);
        }
        if (auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            limit_precision(attention->projection, input);
            limit_precision(attention->output, output, false);
        } else {
            auto& gdn = std::get<GdnParameters>(block.mixer);
            limit_precision(gdn.projection, input);
            limit_precision(gdn.output, output, false);
        }
    }
    return out;
}

} // namespace infernix::models::qwen3_5::execution
