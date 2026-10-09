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
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(input(id)); }
        return with_context(model_.weight(ids.front()).name,
                            [&] { return ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows)); });
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
        out.patch_embedding      = linear({w.patch_embedding});
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
                                                                 linear({layer.output}),
                                                                 linear({layer.fc1}),
                                                                 linear({layer.fc2}),
                                                                 tensor(layer.output_bias),
                                                                 tensor(layer.fc1_bias),
                                                                 tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear({w.merger_fc1});
        out.merger_fc2      = linear({w.merger_fc2});
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
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
    token_embedding = with_context(embedding.name, [&] { return native_weight(embedding.view); });
    const auto& head = source.weight(w.output_head);
    if (!head.view.parts.empty() && head.view.parts.front().parent->geometry.format == QType::Q8_G32_FP16) {
        output_head_q8 = prepare.linear({w.output_head});
    } else {
        output_head = prepare.tensor(w.output_head);
    }
    final_mixer     = prepare.hc(w.final_mixer);

    const auto banks = source.expert_banks();
    if (banks.size() != w.layers.size()) { throw std::logic_error("Qwen4Exp: one expert bank per layer"); }
    const std::size_t per_layer = sizeof(ops::offloaded_moe::ExpertScales) * c.moe.experts;
    scales_ = DeviceBuffer(per_layer * banks.size());
    for (std::size_t i = 0; i < banks.size(); ++i) {
        scales_.copy_from_host(banks[i].scales.data(), per_layer, per_layer * i);
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
            out.mixer     = GdnParameters{prepare.linear({g.query, g.key, g.value, g.z}),
                                          prepare.linear({g.a_projection, g.b_projection}),
                                          prepare.tensor(g.a_log),
                                          prepare.tensor(g.dt_bias),
                                          prepare.tensor(g.convolution),
                                          prepare.tensor(g.norm),
                                          prepare.linear({g.output})};
        }
        out.moe.router         = prepare.tensor(layer.moe.router);
        out.moe.shared_score   = prepare.tensor(layer.moe.shared_score);
        out.moe.shared_gate_up = prepare.linear({layer.moe.shared_gate, layer.moe.shared_up});
        out.moe.shared_down    = prepare.linear({layer.moe.shared_down});
        out.moe.bank           = &banks[i];
        out.moe.device_scales  = reinterpret_cast<const ops::offloaded_moe::ExpertScales*>(
            static_cast<const std::byte*>(scales_.p) + per_layer * i);
        if (layer.ple) {
            const auto& p = *layer.ple;
            out.ple       = PleParameters{prepare.linear({p.key_projection, p.value_projection}),
                                          prepare.tensor(p.key_norm),
                                          prepare.tensor(p.query_norm),
                                          prepare.tensor(p.conv_norm),
                                          prepare.tensor(p.convolution),
                                          prepare.tensor(p.ngram_scale)};
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
    if (w.proposal) {
        draft_head.rows      = prepare.linear({w.proposal->head});
        draft_head.token_ids = prepare.tensor(w.proposal->token_ids);
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
