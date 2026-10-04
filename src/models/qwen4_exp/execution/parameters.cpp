#include "models/qwen4_exp/execution/parameters.h"

#include "core/weight_view.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::execution {
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
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(input(id)); }
        return with_context(model_.weight(*ids.begin()).name,
                            [&] { return ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows)); });
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

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(source);
    const auto& w = source.weights();
    const auto& c = source.config().text;
    const auto& embedding = source.weight(w.token_embedding);
    token_embedding = with_context(embedding.name, [&] { return native_weight(embedding.view); });
    output_head     = prepare.tensor(w.output_head);
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
            out.mixer = AttentionParameters{
                prepare.linear({a->query, a->gate, a->key, a->value, a->index_query, a->index_key}),
                prepare.tensor(a->query_norm), prepare.tensor(a->key_norm),
                prepare.tensor(a->index_query_norm), prepare.tensor(a->index_key_norm),
                prepare.linear({a->output})};
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
}

Parameters::~Parameters() = default;

} // namespace ninfer::models::qwen4_exp::execution
