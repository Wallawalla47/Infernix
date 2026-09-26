#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& config, const TextConfig& target,
                          const LoadOptions& options) {
    // Overlay keeps the whole vision tower in pinned host RAM; resident keeps it on device.
    const artifact::Residency residency =
        options.overlay_vision() ? artifact::Residency::HostPinned
                                 : artifact::Residency::Device;
    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    VisionWeights out;
    out.patch_embedding =
        b.parameter("vision/patch_embedding", {h, config.patch_width()}, {"vision/patch_input"},
                    {}, residency);
    out.patch_embedding_bias = b.direct("vision/patch_embedding_bias", {h}, QType::BF16, residency);
    out.position_embedding = b.direct("vision/position_embedding",
                                      {config.num_position_embeddings, h}, QType::BF16, residency);
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {b.direct(p + "norm1_weight", {h}, QType::BF16, residency),
                             b.direct(p + "norm1_bias", {h}, QType::BF16, residency)};
        layer.norm2       = {b.direct(p + "norm2_weight", {h}, QType::BF16, residency),
                             b.direct(p + "norm2_bias", {h}, QType::BF16, residency)};
        layer.query       = b.parameter(p + "attention/query", {h, h}, {p + "attention_input"},
                                        {}, residency);
        layer.key         = b.parameter(p + "attention/key", {h, h}, {p + "attention_input"},
                                        {}, residency);
        layer.value       = b.parameter(p + "attention/value", {h, h}, {p + "attention_input"},
                                        {}, residency);
        layer.query_bias  = b.direct(p + "attention/query_bias", {h}, QType::BF16, residency);
        layer.key_bias    = b.direct(p + "attention/key_bias", {h}, QType::BF16, residency);
        layer.value_bias  = b.direct(p + "attention/value_bias", {h}, QType::BF16, residency);
        layer.output      = b.parameter(p + "attention/output", {h, h}, {p + "attention_output"},
                                        {}, residency);
        layer.output_bias = b.direct(p + "attention/output_bias", {h}, QType::BF16, residency);
        layer.fc1         = b.parameter(p + "mlp/fc1", {intermediate, h}, {p + "mlp_input"},
                                        {}, residency);
        layer.fc1_bias    = b.direct(p + "mlp/fc1_bias", {intermediate}, QType::BF16, residency);
        layer.fc2         = b.parameter(p + "mlp/fc2", {h, intermediate}, {p + "mlp_activation"},
                                        {}, residency);
        layer.fc2_bias    = b.direct(p + "mlp/fc2_bias", {h}, QType::BF16, residency);
        out.layers.push_back(layer);
    }
    const auto merger = config.merger_width();
    out.merger_norm   = {b.direct("vision/merger/norm_weight", {h}, QType::BF16, residency),
                         b.direct("vision/merger/norm_bias", {h}, QType::BF16, residency)};
    out.merger_fc1    = b.parameter("vision/merger/fc1", {merger, merger}, {"vision/merger/input"},
                                    {}, residency);
    out.merger_fc1_bias = b.direct("vision/merger/fc1_bias", {merger}, QType::BF16, residency);
    out.merger_fc2      = b.parameter("vision/merger/fc2", {target.hidden_size, merger},
                                      {"vision/merger/activation"}, {}, residency);
    out.merger_fc2_bias = b.direct("vision/merger/fc2_bias", {target.hidden_size}, QType::BF16,
                                   residency);
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
