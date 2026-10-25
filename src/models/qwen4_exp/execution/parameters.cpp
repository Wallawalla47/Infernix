#include "models/qwen4_exp/execution/parameters.h"

#include "core/weight_view.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::execution {
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

    ops::WeightInput input(WeightId id) const {
        const auto& bound = model_.weight(id);
        return {bound.view, bound.policy, std::nullopt};
    }

    // One projection over consecutive parameters that share an input; the converter packs each
    // such group into one parent, and the rows keep this order.
    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        return linear(std::span<const WeightId>(ids.begin(), ids.size()));
    }

    LinearParameters linear(std::span<const WeightId> ids) const {
        if (ggml(ids)) {
            return with_context(model_.weight(ids.front()).name, [&] { return ggml_linear(ids); });
        }
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(input(id)); }
        return with_context(model_.weight(ids.front()).name, [&] {
            const auto single = ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows));
            LinearParameters out;
            out.weight = single.weight;
            out.policy = single.policy;
            return out;
        });
    }

    // A native single-parent projection in the shared tower's type (the Vision tower).
    ops::SingleProjectionWeight native(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(input(id)); }
        return with_context(model_.weight(*ids.begin()).name,
                            [&] { return ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows)); });
    }

    // Whether the parameters are stored as GGUF blocks (ggml_rows_v1, ggml_tiles16_v1); a group
    // mixing them with native formats is refused.
    bool ggml(std::span<const WeightId> ids) const {
        bool any = false, all = true;
        for (const auto id : ids) {
            for (const auto& part : model_.weight(id).view.parts) {
                const bool g = is_ggml_matrix_layout(part.parent->geometry.layout);
                any = any || g;
                all = all && g;
            }
        }
        if (any && !all) {
            throw std::invalid_argument(model_.weight(ids.front()).name + ": a projection mixes GGUF blocks with native formats");
        }
        return any;
    }

    // The projection's rows in order: one segment per contiguous region of a GGUF parent.
    LinearParameters ggml_linear(std::span<const WeightId> ids) const {
        LinearParameters out;
        std::int32_t row = 0;
        std::uint64_t k  = 0;
        for (const auto id : ids) {
            const auto& view = model_.weight(id).view;
            if (view.shape.size() != 2) { throw std::invalid_argument("a GGUF projection parameter must be a matrix"); }
            if (k == 0) { k = view.shape[1]; }
            if (view.shape[1] != k) { throw std::invalid_argument("a GGUF projection's parameters must share K"); }
            for (const auto& part : view.parts) {
                const auto& parent = *part.parent;
                const auto& g      = parent.geometry;
                if (g.shape[1] != k || part.begin % k || part.end % k) {
                    throw std::invalid_argument("a GGUF projection region must be whole rows");
                }
                const std::byte* base = parent.device != nullptr ? parent.device : parent.data;
                if (base == nullptr) { throw std::invalid_argument("a GGUF projection needs device-visible weights"); }
                // A tiled parent's region starts on a tile (tile i at 16 i row bytes, as row 16 i of
                // ggml_rows_v1), and only the parent's last tile may be partial.
                const bool tiles = g.layout == QuantLayout::GgmlTiles16;
                if (tiles && (part.begin / k % 16 != 0 || (part.end != g.shape[0] * k && part.end / k % 16 != 0))) {
                    throw std::invalid_argument("a ggml_tiles16_v1 projection region must be whole tiles of 16 rows");
                }
                ops::GgmlSegment s;
                s.format     = g.format;
                s.data       = reinterpret_cast<const std::uint8_t*>(base + part.begin / k * g.code_bytes_per_row);
                s.rows       = static_cast<std::int32_t>((part.end - part.begin) / k);
                s.out_offset = row;
                s.layout     = tiles ? ops::GgmlLayout::Tiles16 : ops::GgmlLayout::Rows;
                row += s.rows;
                out.ggml.push_back(s);
            }
        }
        out.weight.qtype = out.ggml.front().format;
        out.weight.n     = row;
        out.weight.k     = static_cast<std::int32_t>(k);
        out.policy       = ops::LinearPolicy::A16Only;
        return out;
    }

    // The converter packs the six input projections into one parent, unless the index projections
    // are stored in another format (recipe C: FP8 query/gate/key/value, BF16 indexer); then they form
    // two parents and run as two projections.
    AttentionParameters attention(const AttentionWeights& a) const {
        const bool joined = one_parent({a.query, a.gate, a.key, a.value, a.index_query, a.index_key});
        AttentionParameters out{joined ? linear({a.query, a.gate, a.key, a.value, a.index_query, a.index_key})
                                       : linear({a.query, a.gate, a.key, a.value}),
                                std::nullopt,
                                tensor(a.query_norm),
                                tensor(a.key_norm),
                                tensor(a.index_query_norm),
                                tensor(a.index_key_norm),
                                linear({a.output})};
        if (!joined) { out.index_projection = linear({a.index_query, a.index_key}); }
        return out;
    }

    bool one_parent(std::initializer_list<WeightId> ids) const {
        const void* parent = nullptr;
        for (const auto id : ids) {
            for (const auto& part : model_.weight(id).view.parts) {
                if (parent == nullptr) { parent = part.parent; }
                if (part.parent != parent) { return false; }
            }
        }
        return true;
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) { throw std::invalid_argument("direct parameter exceeds Tensor rank"); }
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

    HyperConnectionParameters hc(const HyperConnectionWeights& w) const {
        HyperConnectionParameters out;
        out.norm    = tensor(w.norm);
        out.combine = w.inject.valid();
        out.down    = out.combine ? linear({w.down, w.inject}) : linear({w.down});
        out.up      = linear({w.up});
        return out;
    }

    // The Vision tower's parameters in the shared tower's types (qwen3_5/execution/vision_tower.h),
    // prepared as Qwen3.5 prepares them: fused QKV over one input and its joined bias.
    qwen3_5::execution::VisionParameters vision(const VisionWeights& w) const {
        using qwen3_5::execution::NormParameters;
        const auto norm = [&](const VisionNormWeights& n) { return NormParameters{tensor(n.weight), tensor(n.bias)}; };
        qwen3_5::execution::VisionParameters out;
        out.patch_embedding      = native({w.patch_embedding});
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{input(layer.query), input(layer.key), input(layer.value)};
                return qwen3_5::execution::VisionBlockParameters{norm(layer.norm1),
                                                                 norm(layer.norm2),
                                                                 ops::prepare_linear_weight(qkv),
                                                                 joined_bias({layer.query_bias, layer.key_bias, layer.value_bias}),
                                                                 native({layer.output}),
                                                                 native({layer.fc1}),
                                                                 native({layer.fc2}),
                                                                 tensor(layer.output_bias),
                                                                 tensor(layer.fc1_bias),
                                                                 tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = native({w.merger_fc1});
        out.merger_fc2      = native({w.merger_fc2});
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    // The DFlash2 drafter (docs/maintainer/dflash.md): native Q8 projections, the query block's
    // fused Q/K/V parent, BF16 convolution and selector parameters.
    DFlash2Parameters dflash2(const DFlash2Weights& w, const DFlash2Config& d) const {
        DFlash2Parameters out;
        // Full-head 1-D RoPE at absolute positions (no YaRN: the drafter's own unscaled rotation).
        out.rope               = ops::prepare_rope(static_cast<int>(d.attention.head_dim), d.rope_theta);
        out.feature_projection = linear({w.feature_projection});
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("layers/" + std::to_string(i), [&] {
                const auto& l = w.layers[i];
                DFlash2LayerParameters layer;
                layer.input_norm           = tensor(l.input_norm);
                layer.post_attention_norm  = tensor(l.post_attention_norm);
                layer.query_key_value      = ops::prepare_attn_input_proj_weights(input(l.query), input(l.key),
                                                                                  input(l.value)).weight;
                layer.context_key          = linear({l.context_key});
                layer.context_value        = linear({l.context_value});
                layer.query_norm           = tensor(l.query_norm);
                layer.key_norm             = tensor(l.key_norm);
                layer.output               = linear({l.output});
                layer.gate_up              = linear({l.gate, l.up});
                layer.down                 = linear({l.down});
                layer.attention_base       = tensor(l.attention_base);
                layer.attention_projection = linear({l.attention_projection});
                layer.mlp_base             = tensor(l.mlp_base);
                layer.mlp_projection       = linear({l.mlp_projection});
                return layer;
            }));
        }
        out.selector_hidden      = linear({w.selector_hidden});
        out.predecessor_codebook = tensor(w.predecessor_codebook);
        out.successor_codebook   = tensor(w.successor_codebook);
        return out;
    }

private:
    // One BF16 vector over three consecutive bias parameters (the converter stores them adjacent).
    Tensor joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& part_view = model_.weight(id).view;
            count += weight_element_count(part_view.shape);
            for (const auto& part : part_view.parts) {
                if (!view.parts.empty() &&
                    (view.parts.back().parent != part.parent || view.parts.back().end != part.begin)) {
                    throw std::invalid_argument("Vision QKV bias: the three biases must be one contiguous bank");
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

    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(source);
    const auto& w = source.weights();
    const auto& c = source.config().text;
    const auto& embedding = source.weight(w.token_embedding);
    token_embedding = with_context(embedding.name, [&] { return device_native_weight(embedding.view); });
    if (token_embedding.qtype != QType::BF16 &&
        !(token_embedding.layout == QuantLayout::GgmlRows && token_embedding.qtype == QType::GGML_IQ4_XS)) {
        throw std::invalid_argument(embedding.name + ": the token embedding must be BF16 or IQ4_XS rows");
    }
    const auto& head = source.weight(w.output_head);
    if (!head.view.parts.empty() && (head.view.parts.front().parent->geometry.format == QType::Q8_G32_FP16 ||
                                     is_ggml_matrix_layout(head.view.parts.front().parent->geometry.layout))) {
        output_head_q8 = prepare.linear({w.output_head});
    } else {
        output_head = prepare.tensor(w.output_head);
    }
    if (w.output_head_screen.valid()) { output_head_screen = prepare.linear({w.output_head_screen}).weight; }
    final_mixer     = prepare.hc(w.final_mixer);

    const auto banks = source.expert_banks();
    if (banks.size() != w.layers.size()) { throw std::logic_error("Qwen4Exp: one expert bank per layer"); }
    if (c.gdn.output_ggml_tiled) {
        // Input block j of a GDN output projection (one value head) is HF head
        // (j % key_heads) * per_key + j / key_heads (llama.cpp's tiled order).
        const std::uint32_t nv = c.gdn.value_heads, nk = c.gdn.key_heads, per_key = nv / nk;
        std::vector<std::int32_t> perm(nv);
        for (std::uint32_t j = 0; j < nv; ++j) { perm[j] = static_cast<std::int32_t>((j % nk) * per_key + j / nk); }
        gdn_value_perm_ = DeviceBuffer(perm.size() * sizeof(std::int32_t));
        gdn_value_perm_.copy_from_host(perm.data(), perm.size() * sizeof(std::int32_t), 0);
    }
    const std::size_t per_layer = sizeof(ops::offloaded_moe::ExpertScales) * c.moe.experts;
    scales_ = DeviceBuffer(per_layer * banks.size());
    for (std::size_t i = 0; i < banks.size(); ++i) {
        scales_.copy_from_host(banks[i].scales.data(), per_layer, per_layer * i);
    }
    std::vector<ops::offloaded_moe::LosslessExpertDesign> lossless_design(banks.size());
    if (c.lossless) {
        // The weights' own multipliers (an NVFP4 bank's scale tail); zeros for banks without them. The design
        // reference adds a W4A4 bank's input scales (its declared A4 activations).
        lossless_scales_ = DeviceBuffer(per_layer * banks.size());
        for (std::size_t i = 0; i < banks.size(); ++i) {
            std::vector<ops::offloaded_moe::ExpertScales> scales(c.moe.experts);
            const float* m = banks[i].planes.multipliers;
            if (banks[i].planes.gate_up_format == QType::NVFP4_MUL && m == nullptr) {
                throw std::invalid_argument("Qwen4Exp lossless: an NVFP4 bank without its multipliers");
            }
            const bool a4 = c.design_reference && banks[i].activation == ops::offloaded_moe::ExpertActivation::kA4;
            if (a4 && (banks[i].planes.gate_up_format != QType::NVFP4_MUL || banks[i].scales.size() != c.moe.experts)) {
                throw std::invalid_argument("Qwen4Exp design reference: a W4A4 bank needs NVFP4 records and its input scales");
            }
            bool separate_up = false;
            for (std::uint32_t e = 0; m != nullptr && e < c.moe.experts; ++e) {
                const ops::offloaded_moe::ExpertScales in = a4 ? banks[i].scales[e] : ops::offloaded_moe::ExpertScales{};
                scales[e] = {in.input_gate, in.input_up, in.input_down, m[3 * std::size_t(e)], m[3 * std::size_t(e) + 1],
                             m[3 * std::size_t(e) + 2]};
                separate_up = separate_up || in.input_up != in.input_gate;
            }
            lossless_design[i] = !a4          ? ops::offloaded_moe::LosslessExpertDesign::kNone
                                 : separate_up ? ops::offloaded_moe::LosslessExpertDesign::kA4SeparateUp
                                               : ops::offloaded_moe::LosslessExpertDesign::kA4;
            lossless_scales_.copy_from_host(scales.data(), per_layer, per_layer * i);
        }
    }

    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& layer = w.layers[i];
        BlockParameters out;
        out.attn_hc = prepare.hc(layer.attn_hc);
        out.mlp_hc  = prepare.hc(layer.mlp_hc);
        if (const auto* a = std::get_if<AttentionWeights>(&layer.mixer)) {
            out.mixer = prepare.attention(*a);
        } else {
            const auto& g = std::get<GdnWeights>(layer.mixer);
            GdnParameters gdn{prepare.linear({g.query, g.key, g.value, g.z}),
                              prepare.linear({g.a_projection, g.b_projection}),
                              prepare.tensor(g.a_log),
                              prepare.tensor(g.dt_bias),
                              prepare.tensor(g.convolution),
                              prepare.tensor(g.norm),
                              prepare.linear({g.output})};
            if (c.gdn.output_ggml_tiled) {
                if (!gdn.output.is_ggml()) {
                    throw std::invalid_argument("Qwen4Exp: a ggml-tiled GDN output projection must be GGUF blocks");
                }
                gdn.output.input_perm = static_cast<const std::int32_t*>(gdn_value_perm_.p);
                gdn.output.perm_block = static_cast<std::int32_t>(c.gdn.value_head_dim);
            }
            out.mixer = std::move(gdn);
        }
        out.moe.router         = prepare.tensor(layer.moe.router);
        out.moe.shared_score   = prepare.tensor(layer.moe.shared_score);
        out.moe.shared_gate_up = prepare.linear({layer.moe.shared_gate, layer.moe.shared_up});
        out.moe.shared_down    = prepare.linear({layer.moe.shared_down});
        out.moe.bank           = &banks[i];
        out.moe.device_scales  = reinterpret_cast<const ops::offloaded_moe::ExpertScales*>(
            static_cast<const std::byte*>(scales_.p) + per_layer * i);
        if (c.lossless) {
            out.moe.lossless_scales = reinterpret_cast<const ops::offloaded_moe::ExpertScales*>(
                static_cast<const std::byte*>(lossless_scales_.p) + per_layer * i);
            out.moe.lossless_design = lossless_design[i];
        }
        if (layer.ple) {
            const auto& p = *layer.ple;
            out.ple       = PleParameters{prepare.linear({p.key_projection, p.value_projection}),
                                          prepare.tensor(p.key_norm),
                                          prepare.tensor(p.query_norm),
                                          prepare.tensor(p.conv_norm),
                                          prepare.tensor(p.convolution),
                                          p.ngram_scale.valid() ? prepare.tensor(p.ngram_scale) : Tensor{}};
        }
        layers.push_back(std::move(out));
    }
    if (w.mtp) {
        const auto& m = *w.mtp;
        MtpParameters out;
        out.embedding_norm       = prepare.tensor(m.embedding_norm);
        out.hidden_norm          = prepare.tensor(m.hidden_norm);
        out.embedding_projection = prepare.linear({m.embedding_projection});
        out.hidden_projection    = prepare.linear({m.hidden_projection});
        out.attn_hc              = prepare.hc(m.attn_hc);
        out.mlp_hc               = prepare.hc(m.mlp_hc);
        out.final_mixer          = prepare.hc(m.final_mixer);
        out.attention            = prepare.attention(m.attention);
        out.router               = prepare.tensor(m.router);
        out.shared_score         = prepare.tensor(m.shared_score);
        out.shared_gate_up       = prepare.linear({m.shared_gate, m.shared_up});
        out.shared_down          = prepare.linear({m.shared_down});
        out.experts_gate_up      = prepare.linear(m.expert_gate_up).weight;
        out.experts_down         = prepare.linear(m.expert_down).weight;
        mtp = std::move(out);
    }
    if (w.dflash2) {
        dflash2 = with_context("dflash2", [&] { return prepare.dflash2(*w.dflash2, *model.config().dflash2); });
    }
    if (w.proposal) {
        draft_head.rows      = prepare.linear({w.proposal->head});
        draft_head.token_ids = prepare.tensor(w.proposal->token_ids);
        if (w.proposal->screen.valid()) { draft_head.screen = prepare.linear({w.proposal->screen}).weight; }
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
        if (vision->merger_fc2.weight.n != static_cast<std::int32_t>(model.config().text.hidden_size)) {
            throw std::invalid_argument("Qwen4Exp vision: the merger's output width differs from the text hidden size");
        }
    }
}

Parameters::~Parameters() = default;

} // namespace infernix::models::qwen4_exp::execution
