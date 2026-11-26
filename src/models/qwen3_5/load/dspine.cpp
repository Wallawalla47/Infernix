#include "models/qwen3_5/load/bindings.h"

// The code bridge of a standalone DSpine drafter (docs/maintainer/dspine.md, "The artifact"): its
// projection [3R, C] (the gate, predecessor-gate and value rows, one below the other), the gate's
// bias [R], each layer's read [C, H] and write [H, R], the last write transposed [R, H], and every
// token's sender rows [V, 2R]. The drafter's layers bind as a DFlash drafter's (bind_draft).

namespace infernix::models::qwen3_5::loading {

void bind_dspine(Bindings& b, DraftWeights& weights, const DraftConfig& config,
                 const TextConfig& target) {
    const auto& extra = config.dspine.value();
    const std::uint64_t h = target.hidden_size, code = extra.code_size, rank = extra.rank;
    DSpineBridgeWeights bridge;
    bridge.projection   = b.parameter("dspine/bridge/projection", {3 * rank, code}, {"dspine/bridge/code_input"});
    bridge.gate_bias    = b.direct("dspine/bridge/gate_bias", {rank});
    bridge.head_write   = b.direct("dspine/bridge/head_write", {rank, h});
    bridge.sender_table = b.direct("dspine/bridge/sender_table", {target.vocab_size, 2 * rank});
    bridge.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = "dspine/bridge/layers/" + std::to_string(i) + "/";
        bridge.layers.push_back({b.parameter(p + "read", {code, h}, {p + "read_input"}),
                                 b.parameter(p + "write", {h, rank}, {p + "message"})});
    }
    weights.dspine = std::move(bridge);
}

std::string dspine_head_input(const artifact::Directory& target) {
    // The target's head through its DFlash2 drafter's use where the artifact declares one (the
    // drafter's final hidden is a masked drafter's), else through the text's own.
    return target.uses.contains({"text/output_head", "dflash2/final_hidden"}) ? "dflash2/final_hidden"
                                                                              : "text/final_hidden";
}

} // namespace infernix::models::qwen3_5::loading
