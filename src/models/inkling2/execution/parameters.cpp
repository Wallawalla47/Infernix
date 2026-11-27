#include "models/inkling2/execution/parameters.h"

#include "core/weight_view.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/weight_input.h"

#include <array>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace infernix::models::inkling2 {
namespace {

// A direct parameter's words as device code reads them (BF16 or FP32, contiguous): in device memory, or in pinned
// host memory at its mapped address (--embedding-host's token embedding), which may differ from its host address.
template <class T>
const T* words(const Model& model, WeightId id) {
    const BoundWeight& w     = model.weight(id);
    const WeightRegion region = contiguous_weight_region(w.view);
    const WeightGeometry& g  = region.parent->geometry;
    const QType want         = sizeof(T) == 4 ? QType::FP32 : QType::BF16;
    if (g.layout != QuantLayout::Contiguous || g.format != want || region.parent->device == nullptr) {
        throw std::invalid_argument(w.name + ": a device-readable contiguous " + (sizeof(T) == 4 ? "FP32" : "BF16") +
                                    " tensor is needed");
    }
    return reinterpret_cast<const T*>(region.parent->device + region.begin * sizeof(T));
}

const std::uint16_t* bf16(const Model& model, WeightId id) { return words<std::uint16_t>(model, id); }
const float* fp32(const Model& model, WeightId id) { return words<float>(model, id); }

// The input divisor of a native Weight: an NVFP4 parent's projections run on 16-bit activations here (the DFlash2
// companion's NVFP4 MLP), which read none, but the native ABI validates a positive one (as prepare_linear_weight
// sets it); 0 for every other format.
float a16_divisor(const WeightView& view) {
    const auto region = contiguous_weight_region(view);
    return region.parent != nullptr && region.parent->geometry.format == QType::NVFP4 ? 1.0F : 0.0F;
}

Weight linear(const Model& model, WeightId id) {
    const BoundWeight& w = model.weight(id);
    try {
        return native_weight(w.view, a16_divisor(w.view));
    } catch (const std::exception& e) {
        throw std::invalid_argument(w.name + ": " + e.what());
    }
}

// Row ranges of one parent, each right after the one before, as one matrix.
Weight stacked(const Model& model, std::initializer_list<WeightId> ids) {
    const WeightView& first = model.weight(*ids.begin()).view;
    WeightView view;
    view.shape = {0, first.shape.size() == 2 ? first.shape[1] : 0};
    std::string names;
    for (const WeightId id : ids) {
        const WeightView& part = model.weight(id).view;
        if (part.shape.size() != 2 || part.shape[1] != view.shape[1]) {
            throw std::invalid_argument(model.weight(id).name + ": stacked matrices need equal K");
        }
        view.shape[0] += part.shape[0];
        view.parts.insert(view.parts.end(), part.parts.begin(), part.parts.end());
        names += (names.empty() ? "" : " + ") + model.weight(id).name;
    }
    try {
        return native_weight(view, a16_divisor(view));
    } catch (const std::exception& e) {
        throw std::invalid_argument(names + ": not adjacent rows of one parent (" + e.what() + ")");
    }
}

Weight stacked(const Model& model, WeightId first, WeightId second) { return stacked(model, {first, second}); }

// BF16 vectors stored one right after another (one parent), as one pointer to the first.
const std::uint16_t* adjacent_bf16(const Model& model, std::initializer_list<WeightId> ids) {
    const std::uint16_t* first = nullptr;
    std::size_t next           = 0;
    for (const WeightId id : ids) {
        const std::uint16_t* p = bf16(model, id);
        const auto& shape      = model.weight(id).view.shape;
        if (first == nullptr) {
            first = p;
        } else if (p != first + next) {
            throw std::invalid_argument(model.weight(id).name + ": not stored right after the vector before it");
        }
        next += shape.empty() ? 0 : static_cast<std::size_t>(shape[0]);
    }
    return first;
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

// A direct parameter as a Tensor of its stored shape (innermost axis first).
Tensor tensor(const Model& model, WeightId id) {
    const BoundWeight& w = model.weight(id);
    const auto& view     = w.view;
    if (view.shape.empty() || view.shape.size() > 4) { throw std::invalid_argument(w.name + ": a direct tensor of rank 1-4"); }
    std::array<std::int32_t, 4> axes{1, 1, 1, 1};
    for (std::size_t i = 0; i < view.shape.size(); ++i) {
        axes[i] = static_cast<std::int32_t>(view.shape[view.shape.size() - 1 - i]);
    }
    try {
        return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
    } catch (const std::exception& e) {
        throw std::invalid_argument(w.name + ": " + e.what());
    }
}

DFlash2Operands dflash2_operands(const Model& model, const DFlash2Weights& w, const DFlash2Config& c) {
    DFlash2Operands out;
    // Full-head 1-D RoPE at absolute positions (the drafter's own theta, unscaled).
    out.rope               = ops::prepare_rope(static_cast<int>(c.head_dim), c.rope_theta);
    out.feature_projection = linear(model, w.feature_projection);
    out.context_norm       = tensor(model, w.context_norm);
    out.final_norm         = tensor(model, w.final_norm);
    for (const DFlash2LayerWeights& l : w.layers) {
        DFlash2LayerOperands o;
        o.input_norm          = tensor(model, l.input_norm);
        o.post_attention_norm = tensor(model, l.post_attention_norm);
        const auto input      = [&](WeightId id) {
            const BoundWeight& b = model.weight(id);
            return ops::WeightInput{b.view, b.policy, std::nullopt};
        };
        o.query_key_value      = ops::prepare_attn_input_proj_weights(input(l.query), input(l.key), input(l.value)).weight;
        o.context_key          = linear(model, l.context_key);
        o.context_value        = linear(model, l.context_value);
        o.query_norm           = tensor(model, l.query_norm);
        o.key_norm             = tensor(model, l.key_norm);
        o.output               = linear(model, l.output);
        o.gate_up              = stacked(model, l.gate, l.up);
        o.down                 = linear(model, l.down);
        o.attention_base       = tensor(model, l.attention_base);
        o.mlp_base             = tensor(model, l.mlp_base);
        o.attention_projection = linear(model, l.attention_projection);
        o.mlp_projection       = linear(model, l.mlp_projection);
        out.layers.push_back(o);
    }
    out.selector_hidden      = linear(model, w.selector_hidden);
    out.predecessor_codebook = tensor(model, w.predecessor_codebook);
    out.successor_codebook   = tensor(model, w.successor_codebook);
    return out;
}

VisionOperands vision_operands(const Model& model, const VisionWeights& w) {
    VisionOperands out;
    out.patch_embedding = linear(model, w.patch_embedding);
    out.position_table  = bf16(model, w.position_embedding);
    out.class_row       = bf16(model, w.class_row);
    out.pre_norm        = bf16(model, w.pre_norm);
    out.pre_norm_bias   = bf16(model, w.pre_norm_bias);
    for (const VisionLayerWeights& l : w.layers) {
        VisionLayerOperands o;
        o.attention_norm      = bf16(model, l.attention_norm);
        o.attention_norm_bias = bf16(model, l.attention_norm_bias);
        o.qkv                 = stacked(model, {l.query, l.key, l.value});
        o.qkv_bias            = adjacent_bf16(model, {l.query_bias, l.key_bias, l.value_bias});
        o.output              = linear(model, l.output);
        o.output_bias         = bf16(model, l.output_bias);
        o.attention_scale     = bf16(model, l.attention_scale);
        o.mlp_norm            = bf16(model, l.mlp_norm);
        o.mlp_norm_bias       = bf16(model, l.mlp_norm_bias);
        o.up                  = linear(model, l.up);
        o.up_bias             = bf16(model, l.up_bias);
        o.down                = linear(model, l.down);
        o.down_bias           = bf16(model, l.down_bias);
        o.mlp_scale           = bf16(model, l.mlp_scale);
        out.layers.push_back(o);
    }
    out.projector_in  = linear(model, w.projector_in);
    out.projector_out = linear(model, w.projector_out);
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
    // The default and reproducible routes' kernels compute these scalars as constants (the lossless route reads the
    // config): another value would give those routes another model's arithmetic.
    if (c.rms_norm_eps != k::kRmsEps || c.qk_clip_threshold != k::kQkClipThreshold ||
        c.relative_clip_threshold != k::kRelativeClipThreshold || c.log_scaling_alpha != k::kLogScalingAlpha ||
        static_cast<float>(c.log_scaling_floor) != k::kLogScalingFloor) {
        throw std::invalid_argument("inkling2: the kernels are built for Inkling 2 Mini's rms_norm_eps 1e-6, "
                                    "qk_clip_threshold 16*sqrt(2), relative_clip_threshold 8 and log scaling 0.1 / 8192; "
                                    "the artifact's config states others");
    }
    embedding      = bf16(model, w.token_embedding);
    embedding_norm = bf16(model, w.embedding_norm);
    final_norm     = bf16(model, w.final_norm);
    output_query   = fp32(model, w.output_query);
    head           = linear(model, w.output_head);
    layers.reserve(w.layers.size());
    for (std::uint32_t i = 0; i < w.layers.size(); ++i) { layers.push_back(layer_operands(model, w.layers[i], c, i)); }
    if (w.dspark) { dspark = dspark_operands(model, *w.dspark, *model.config().dspark); }
    if (w.dflash2) { dflash2 = dflash2_operands(model, *w.dflash2, *model.config().dflash2); }
    if (w.vision) { vision = vision_operands(model, *w.vision); }
    lora   = prepare_lora(model.lora(), c, model.options().lossless);
    ablate = prepare_ablate(model.ablate(), c.num_hidden_layers, c.hidden_size);
}

} // namespace infernix::models::inkling2
