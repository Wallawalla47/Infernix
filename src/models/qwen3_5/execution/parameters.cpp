#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
    explicit Prepare(const Model& model) : model_(model) {}

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

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
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
        return with_context(model_.weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(
                model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                model_.input(moe.shared.gate), model_.input(moe.shared.up),
                model_.input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
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
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    const auto& config   = model.config().text;
    if (config.rope_parameters) {
        const auto& rope = *config.rope_parameters;
        text.rope = ops::prepare_rope(dimension(rope.rotary_dim), rope.rope_theta,
                                     {rope.yarn_factor, config.max_position_embeddings});
    }
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
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
    if (column_invariant_text_supported(text)) {
        invariant_text = column_invariant_text(text);
        constexpr std::pair<bool, bool> kModes[] = {{false, false}, {true, false}, {false, true}};
        for (const auto [a8, a8_residual] : kModes) {
            for (const auto [a4, a4_residual] : kModes) {
                const DecodePrecision precision{a8, a4, a8_residual, a4_residual};
                precision_text[precision_index(precision)] =
                    precision_decode_text(text, precision);
            }
        }
    }
}

namespace {

bool fp8(const LinearParameters& p) { return p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16; }
bool nvfp4_or_fp8(const LinearParameters& p) { return fp8(p) || p.weight.qtype == QType::NVFP4; }

bool single_nvfp4_or_fp8(const ops::ProjectionWeights& projection) {
    const auto* single = std::get_if<ops::SingleProjectionWeight>(&projection);
    return single != nullptr && nvfp4_or_fp8(*single);
}

// The vocabulary head: FP8, or Q8 at the 27B's [248320, 5120].
bool invariant_head(const LinearParameters& p) {
    return fp8(p) || (p.weight.qtype == QType::Q8_G32_FP16 && p.weight.n == 248320 &&
                      p.weight.k == 5120);
}

void make_invariant(LinearParameters& p) { p.policy = ops::LinearPolicy::A16ColumnInvariant; }

// Keeps the projection's low-precision permission only where its format's class is permitted; a
// projection permitted its class's one-term activations may take its residual profile instead
// (the NVFP4 one, `a4_residual_here`, is registered for the MLP only).
void limit_precision(LinearParameters& p, DecodePrecision precision, bool a4_residual_here) {
    const bool fp8   = p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const bool nvfp4 = p.weight.qtype == QType::NVFP4;
    if ((fp8 && precision.a8) || (nvfp4 && precision.a4)) { return; }
    if (fp8 && precision.a8_residual && ops::allows_a8(p.policy)) {
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

void make_invariant(ops::ProjectionWeights& projection) {
    if (auto* single = std::get_if<ops::SingleProjectionWeight>(&projection)) {
        make_invariant(*single);
    }
}

} // namespace

bool column_invariant_text_supported(const TextParameters& text) {
    if (!invariant_head(text.output_head)) { return false; }
    for (const BlockParameters& block : text.layers) {
        const auto* dense = std::get_if<DenseParameters>(&block.ffn);
        if (dense == nullptr || !nvfp4_or_fp8(dense->gate_up) || !nvfp4_or_fp8(dense->down)) {
            return false;
        }
        if (const auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            if (!single_nvfp4_or_fp8(attention->projection) || !nvfp4_or_fp8(attention->output)) {
                return false;
            }
        } else {
            const auto& gdn = std::get<GdnParameters>(block.mixer);
            if (!single_nvfp4_or_fp8(gdn.projection) || !nvfp4_or_fp8(gdn.output)) {
                return false;
            }
            // The control's column-invariant schedule is the 27B's [96, 5120] BF16 parent.
            const auto* control = std::get_if<LinearParameters>(&gdn.control);
            if (control == nullptr || control->weight.qtype != QType::BF16 ||
                control->weight.n != 96 || control->weight.k != 5120) {
                return false;
            }
        }
    }
    return true;
}

TextParameters column_invariant_text(const TextParameters& text) {
    TextParameters out = text;
    make_invariant(out.output_head);
    for (BlockParameters& block : out.layers) {
        auto& dense = std::get<DenseParameters>(block.ffn);
        make_invariant(dense.gate_up);
        make_invariant(dense.down);
        if (auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            make_invariant(attention->projection);
            make_invariant(attention->output);
        } else {
            auto& gdn = std::get<GdnParameters>(block.mixer);
            make_invariant(gdn.projection);
            make_invariant(gdn.output);
            if (auto* control = std::get_if<LinearParameters>(&gdn.control)) {
                make_invariant(*control);
            }
        }
    }
    return out;
}

TextParameters precision_decode_text(const TextParameters& text, DecodePrecision precision) {
    TextParameters out = text;
    // The vocabulary head keeps 16-bit activations unless its class's one-term profile is on.
    limit_precision(out.output_head, {precision.a8, precision.a4, false, false}, false);
    for (BlockParameters& block : out.layers) {
        auto& dense = std::get<DenseParameters>(block.ffn);
        limit_precision(dense.gate_up, precision, true);
        limit_precision(dense.down, precision, true);
        if (auto* attention = std::get_if<AttentionParameters>(&block.mixer)) {
            limit_precision(attention->projection, precision);
            limit_precision(attention->output, precision, false);
        } else {
            auto& gdn = std::get<GdnParameters>(block.mixer);
            limit_precision(gdn.projection, precision);
            limit_precision(gdn.output, precision, false);
        }
    }
    return out;
}

} // namespace infernix::models::qwen3_5::execution
