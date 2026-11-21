#include "models/inkling2/execution/parameters.h"

#include "core/weight_view.h"

#include <stdexcept>
#include <string>

namespace infernix::models::inkling2 {
namespace {

// A direct parameter's device words (BF16 or FP32, contiguous and resident).
template <class T>
const T* words(const Model& model, WeightId id) {
    const BoundWeight& w     = model.weight(id);
    const WeightRegion region = contiguous_weight_region(w.view);
    const WeightGeometry& g  = region.parent->geometry;
    const QType want         = sizeof(T) == 4 ? QType::FP32 : QType::BF16;
    if (g.layout != QuantLayout::Contiguous || g.format != want || region.parent->data == nullptr) {
        throw std::invalid_argument(w.name + ": a resident contiguous " + (sizeof(T) == 4 ? "FP32" : "BF16") +
                                    " tensor is needed");
    }
    return reinterpret_cast<const T*>(region.parent->data + region.begin * sizeof(T));
}

const std::uint16_t* bf16(const Model& model, WeightId id) { return words<std::uint16_t>(model, id); }
const float* fp32(const Model& model, WeightId id) { return words<float>(model, id); }

Weight linear(const Model& model, WeightId id) {
    const BoundWeight& w = model.weight(id);
    try {
        return native_weight(w.view);
    } catch (const std::exception& e) {
        throw std::invalid_argument(w.name + ": " + e.what());
    }
}

// Two row ranges of one parent, the second right after the first, as one matrix.
Weight stacked(const Model& model, WeightId first, WeightId second) {
    const WeightView& a = model.weight(first).view;
    const WeightView& b = model.weight(second).view;
    if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[1]) {
        throw std::invalid_argument(model.weight(first).name + ": stacked matrices need equal K");
    }
    WeightView view;
    view.shape = {a.shape[0] + b.shape[0], a.shape[1]};
    view.parts = a.parts;
    view.parts.insert(view.parts.end(), b.parts.begin(), b.parts.end());
    try {
        return native_weight(view);
    } catch (const std::exception& e) {
        throw std::invalid_argument(model.weight(first).name + " + " + model.weight(second).name +
                                    ": not adjacent rows of one parent (" + e.what() + ")");
    }
}

ops::w4a4_moe::Bank bank(const Model& model, const MoeWeights& w, const TextConfig& c, ExpertFormat format) {
    const BoundWeight& experts = model.weight(w.experts);
    if (experts.view.parts.size() != 1 || experts.view.parts[0].parent == nullptr) {
        throw std::invalid_argument(experts.name + ": an expert bank is one complete parent");
    }
    const ExpertBankPlanes planes = expert_bank_planes(*experts.view.parts[0].parent);
    if (planes.records == nullptr || planes.experts != c.experts || planes.hidden != c.hidden_size ||
        planes.intermediate != c.intermediate) {
        throw std::invalid_argument(experts.name + ": a resident bank of the config's geometry is needed");
    }
    ops::w4a4_moe::Bank b;
    b.records       = reinterpret_cast<const std::uint8_t*>(planes.records);
    b.record_stride = planes.record_stride;
    b.format        = format == ExpertFormat::Nvfp4FourOverSix ? ops::w4a4_moe::ExpertFormat::Nvfp4FourOverSix
                                                               : ops::w4a4_moe::ExpertFormat::Mxfp8;
    if (format == ExpertFormat::Nvfp4FourOverSix) {
        b.amax         = fp32(model, w.expert_amax);
        b.denominators = fp32(model, w.expert_denominators);
    }
    b.experts      = static_cast<std::int32_t>(c.experts);
    b.hidden       = static_cast<std::int32_t>(c.hidden_size);
    b.intermediate = static_cast<std::int32_t>(c.intermediate);
    b.gate_softcap = c.gate_softcap;
    b.up_softcap   = c.up_softcap;
    return b;
}

LayerOperands layer_operands(const Model& model, const LayerWeights& w, const TextConfig& c, std::uint32_t index) {
    LayerOperands out;
    out.full     = c.layer_types[index] == AttentionKind::Full;
    out.boundary = index % c.residual_block == 0;
    out.format   = c.expert_formats[index] == ExpertFormat::Nvfp4FourOverSix ? ops::w4a4_moe::ExpertFormat::Nvfp4FourOverSix
                                                                             : ops::w4a4_moe::ExpertFormat::Mxfp8;
    if (index > 0) { out.attention_mix_query = fp32(model, w.attention_mix_query); }
    if (out.full) {
        out.norms[0] = bf16(model, w.query_norm);
        out.norms[1] = bf16(model, w.key_norm);
        out.norms[2] = bf16(model, w.value_norm);
    } else {
        out.norms[0] = out.norms[1] = out.norms[2] = bf16(model, w.attention_norm);
    }
    out.mlp_norm       = bf16(model, w.mlp_norm);
    out.mlp_mix_query  = fp32(model, w.mlp_mix_query);
    out.attention_conv = bf16(model, w.attention_conv);
    out.mlp_conv       = bf16(model, w.mlp_conv);
    const AttentionWeights& a = w.attention;
    AttentionOperands& o      = out.attention;
    o.query    = linear(model, a.query);
    o.relative = linear(model, a.relative);
    o.gate     = linear(model, a.gate);
    o.key      = linear(model, a.key);
    o.value    = linear(model, a.value);
    o.sink     = linear(model, a.sink);
    if (out.full) {
        o.key_taps   = linear(model, a.key_taps);
        o.value_taps = linear(model, a.value_taps);
    }
    o.output                = linear(model, a.output);
    o.layer.full            = out.full;
    o.layer.query_scale     = bf16(model, a.query_scale);
    o.layer.relative_scale  = bf16(model, a.relative_scale);
    o.layer.relative_table  = bf16(model, a.relative_table);
    o.layer.key_conv        = bf16(model, a.key_conv);
    o.layer.value_conv      = bf16(model, a.value_conv);
    if (w.messages) {
        out.messages        = true;
        out.message_queries = fp32(model, w.messages->queries);
        for (std::size_t k = 0; k < out.gates.size(); ++k) {
            const MessageGateWeights& g = w.messages->gates[k];
            out.gates[k].down           = linear(model, g.down);
            out.gates[k].forget_gain    = stacked(model, g.forget, g.gain);
        }
    }
    out.router      = bf16(model, w.moe.router);
    out.router_bias = fp32(model, w.moe.router_bias);
    out.bank        = bank(model, w.moe, c, c.expert_formats[index]);
    return out;
}

DSparkOperands dspark_operands(const Model& model, const DSparkWeights& w, const DSparkConfig& c) {
    DSparkOperands out;
    out.feature_weights = bf16(model, w.feature_weights);
    out.mask_embedding  = bf16(model, w.mask_embedding);
    out.final_norm      = bf16(model, w.final_norm);
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const DSparkLayerWeights& l = w.layers[i];
        DSparkLayerOperands o;
        o.sliding             = c.layer_types.at(i) == AttentionKind::Sliding;
        o.context_norm        = bf16(model, l.context_norm);
        o.input_norm          = bf16(model, l.input_norm);
        o.post_attention_norm = bf16(model, l.post_attention_norm);
        o.query               = linear(model, l.query);
        o.key                 = linear(model, l.key);
        o.value               = linear(model, l.value);
        o.context_key         = linear(model, l.context_key);
        o.context_value       = linear(model, l.context_value);
        o.query_norm          = bf16(model, l.query_norm);
        o.key_norm            = bf16(model, l.key_norm);
        o.context_key_norm    = bf16(model, l.context_key_norm);
        o.output              = linear(model, l.output);
        o.gate                = linear(model, l.gate);
        o.up                  = linear(model, l.up);
        o.down                = linear(model, l.down);
        out.layers.push_back(o);
    }
    out.markov_context_norm    = bf16(model, w.markov_context_norm);
    out.markov_embedding       = bf16(model, w.markov_embedding);
    out.markov_projection      = linear(model, w.markov_projection);
    out.markov_projection_bias = bf16(model, w.markov_projection_bias);
    out.fold                   = linear(model, w.fold);
    out.fold_bias              = bf16(model, w.fold_bias);
    return out;
}

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const TextWeights& w = model.weights();
    const TextConfig& c  = model.config().text;
    namespace k        = ops::inkling2;
    const auto is      = [](std::uint32_t value, std::int32_t want) { return value == static_cast<std::uint32_t>(want); };
    if (!is(c.hidden_size, k::kHidden) || !is(c.heads, k::kHeads) || !is(c.kv_heads, k::kKvHeads) ||
        !is(c.head_dim, k::kHeadDim) || !is(c.relative_dim, k::kRelDim) || !is(c.relative_extent, k::kRelativeExtent) ||
        !is(c.sliding_window, k::kWindow) || c.conv_kernel != 4 || !is(c.top_k, ops::w4a4_moe::kTopK) ||
        c.residual_block == 0) {
        throw std::invalid_argument("inkling2: the kernels are built for Inkling 2 Mini's geometry");
    }
    embedding      = bf16(model, w.token_embedding);
    embedding_norm = bf16(model, w.embedding_norm);
    final_norm     = bf16(model, w.final_norm);
    output_query   = fp32(model, w.output_query);
    head           = linear(model, w.output_head);
    layers.reserve(w.layers.size());
    for (std::uint32_t i = 0; i < w.layers.size(); ++i) { layers.push_back(layer_operands(model, w.layers[i], c, i)); }
    if (w.dspark) { dspark = dspark_operands(model, *w.dspark, *model.config().dspark); }
}

} // namespace infernix::models::inkling2
