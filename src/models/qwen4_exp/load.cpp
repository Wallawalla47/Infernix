#include "models/qwen4_exp/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/registry.h"

#include <cmath>
#include <cstring>
#include <map>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Residency;
using artifact::Shape;

ops::LinearPolicy linear_policy(artifact::ActivationPolicy policy) {
    switch (policy) {
    case artifact::ActivationPolicy::A16Only:
        return ops::LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return ops::LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return ops::LinearPolicy::AllowA4;
    }
    throw ArtifactError("unknown activation policy");
}

struct PendingWeight {
    artifact::ParameterReference reference;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

// Logical demands of the selected model; each parameter is declared exactly once.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {},
                       std::optional<QType> exact_format = {},
                       Residency residency                = Residency::Device) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, exact_format);
        if (!input.empty()) {
            const auto& use = binder_.use(name, input);
            if (!use.activation_policy) {
                throw ArtifactError(name + "@" + std::string(input) + ": missing activation policy");
            }
            if (!use.auxiliaries.empty()) {
                throw ArtifactError(name + ": Qwen4Exp weights carry no Use auxiliaries");
            }
            pending.policy = linear_policy(*use.activation_policy);
        }
        const WeightId id{weights.size()};
        ids_.emplace(std::move(name), id);
        weights.push_back(std::move(pending));
        return id;
    }

    WeightId direct(std::string name, Shape shape, QType format = QType::BF16, Residency residency = Residency::Device) {
        return parameter(std::move(name), std::move(shape), {}, format, residency);
    }

    std::vector<PendingWeight> weights;

private:
    artifact::Binder& binder_;
    std::map<std::string, WeightId, std::less<>> ids_;
};

HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& prefix,
                               bool combine) {
    const std::uint64_t width = c.residual_width(), rank = c.hc.rank;
    HyperConnectionWeights out;
    out.norm = b.direct(prefix + "norm", {width});
    out.down = b.parameter(prefix + "down", {rank, width}, prefix + "input");
    out.up   = b.parameter(prefix + "up", {width, rank}, prefix + "mix_activation");
    if (combine) {
        out.inject = b.parameter(prefix + "inject", {c.hc.streams, width}, prefix + "input");
    }
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, q = c.attention.query_width(),
                        k = c.attention.key_width();
    const std::uint64_t iq = std::uint64_t(c.qsa.index_heads) * c.qsa.index_head_dim;
    const std::string input = prefix + "mixer_input", p = prefix + "attention/";
    AttentionWeights out;
    out.query            = b.parameter(p + "query", {q, h}, input);
    out.gate             = b.parameter(p + "gate", {q, h}, input);
    out.key              = b.parameter(p + "key", {k, h}, input);
    out.value            = b.parameter(p + "value", {k, h}, input);
    out.index_query      = b.parameter(p + "index_query", {iq, h}, input);
    out.index_key        = b.parameter(p + "index_key", {c.qsa.index_head_dim, h}, input);
    out.query_norm       = b.direct(p + "query_norm", {c.attention.head_dim});
    out.key_norm         = b.direct(p + "key_norm", {c.attention.head_dim});
    out.index_query_norm = b.direct(p + "index_query_norm", {c.qsa.index_head_dim});
    out.index_key_norm   = b.direct(p + "index_key_norm", {c.qsa.index_head_dim});
    out.output           = b.parameter(p + "output", {h, q}, p + "gated_output");
    return out;
}

GdnWeights bind_gdn(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, kw = c.gdn.key_width(), vw = c.gdn.value_width(),
                        nv = c.gdn.value_heads;
    const std::string input = prefix + "mixer_input", p = prefix + "gdn/";
    GdnWeights out;
    out.query        = b.parameter(p + "query", {kw, h}, input);
    out.key          = b.parameter(p + "key", {kw, h}, input);
    out.value        = b.parameter(p + "value", {vw, h}, input);
    out.z            = b.parameter(p + "z", {vw, h}, input);
    out.a_projection = b.parameter(p + "a_projection", {nv, h}, input);
    out.b_projection = b.parameter(p + "b_projection", {nv, h}, input);
    out.a_log        = b.direct(p + "a_log", {nv}, QType::FP32);
    out.dt_bias      = b.direct(p + "dt_bias", {nv}, QType::FP32);
    out.convolution  = b.direct(p + "convolution", {c.gdn.conv_kernel, c.gdn.conv_channels()});
    out.norm         = b.direct(p + "norm", {c.gdn.value_head_dim});
    out.output       = b.parameter(p + "output", {h, vw}, p + "gated_output");
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix, Residency experts) {
    const std::uint64_t h = c.hidden_size, e = c.moe.experts, s = c.moe.shared_intermediate;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    MoeWeights out;
    out.router       = b.parameter(p + "router", {e, h}, input);
    out.shared_score = b.parameter(p + "shared_score", {1, h}, input);
    out.shared_gate  = b.parameter(p + "shared/gate", {s, h}, input);
    out.shared_up    = b.parameter(p + "shared/up", {s, h}, input);
    out.shared_down  = b.parameter(p + "shared/down", {h, s}, p + "shared/product");
    out.experts      = b.parameter(p + "experts", {e, h, c.moe.intermediate}, input, QType::NVFP4_MUL, experts);
    out.input_scales = b.parameter(p + "expert_input_scales", {e, 3}, {}, QType::FP32,
                                   Residency::Values);
    return out;
}

PleWeights bind_ple(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, width = c.residual_width(), dim = c.ple.embed_dim;
    const std::string p = prefix + "ple/", input = prefix + "ple/embeddings";
    PleWeights out;
    out.key_projection   = b.parameter(p + "key_projection", {width, dim}, input);
    out.value_projection = b.parameter(p + "value_projection", {h, dim}, input);
    out.key_norm         = b.direct(p + "key_norm", {width});
    out.query_norm       = b.direct(p + "query_norm", {width});
    out.conv_norm        = b.direct(p + "conv_norm", {width});
    out.convolution      = b.direct(p + "convolution", {c.ple.conv_kernel, width});
    out.ngram_scale      = b.direct(p + "ngram_scale", {1});
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& c, Residency experts) {
    TextWeights out;
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    out.token_embedding = b.parameter("text/token_embedding", {v, h}, {}, QType::BF16,
                                      Residency::HostPinned);
    out.output_head = b.parameter("text/output_head", {v, h}, "text/final_hidden");
    out.final_mixer = bind_hc(b, c, "text/final_mixer/", false);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        BlockWeights block;
        if (i == c.ple.layer) { block.ple = bind_ple(b, c, prefix); }
        block.attn_hc = bind_hc(b, c, prefix + "attn_hc/", true);
        if (c.layer_types[i] == MixerKind::Attention) {
            block.mixer = bind_attention(b, c, prefix);
        } else {
            block.mixer = bind_gdn(b, c, prefix);
        }
        block.mlp_hc = bind_hc(b, c, prefix + "mlp_hc/", true);
        block.moe    = bind_moe(b, c, prefix, experts);
        out.layers.push_back(std::move(block));
    }
    return out;
}

MtpWeights bind_mtp(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, width = c.residual_width(), e = c.moe.experts,
                        ir = c.moe.intermediate, s = c.moe.shared_intermediate;
    MtpWeights out;
    out.embedding_norm       = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm          = b.direct("mtp/hidden_norm", {width});
    out.embedding_projection = b.parameter("mtp/embedding_projection", {h, h}, "mtp/embedding_input");
    out.hidden_projection    = b.parameter("mtp/hidden_projection", {h, h}, "mtp/hidden_input");
    out.final_mixer          = bind_hc(b, c, "mtp/final_mixer/", false);
    const std::string prefix = "mtp/layers/0/";
    out.attn_hc   = bind_hc(b, c, prefix + "attn_hc/", true);
    out.attention = bind_attention(b, c, prefix);
    out.mlp_hc    = bind_hc(b, c, prefix + "mlp_hc/", true);
    const std::string p = prefix + "moe/", input = prefix + "ffn_input";
    out.router       = b.parameter(p + "router", {e, h}, input);
    out.shared_score = b.parameter(p + "shared_score", {1, h}, input);
    out.shared_gate  = b.parameter(p + "shared/gate", {s, h}, input);
    out.shared_up    = b.parameter(p + "shared/up", {s, h}, input);
    out.shared_down  = b.parameter(p + "shared/down", {h, s}, p + "shared/product");
    for (std::uint64_t i = 0; i < e; ++i) {
        const std::string ep = p + "experts/" + std::to_string(i) + "/";
        out.expert_gate_up.push_back(b.parameter(ep + "gate", {ir, h}, input));
        out.expert_gate_up.push_back(b.parameter(ep + "up", {ir, h}, input));
        out.expert_down.push_back(b.parameter(ep + "down", {h, ir}, ep + "product"));
    }
    return out;
}

// Qwen3.5's tower binding (qwen3_5/load/vision.cpp) with the same parameter names; the merger's
// output width is the text hidden size.
VisionWeights bind_vision(Bindings& b, const qwen3_5::VisionConfig& config, const TextConfig& target,
                          Residency residency) {
    const std::uint64_t h = config.hidden_size, intermediate = config.intermediate_size;
    const auto use = [&](std::string name, Shape shape, std::string_view input) {
        return b.parameter(std::move(name), std::move(shape), input, {}, residency);
    };
    const auto bf16 = [&](std::string name, Shape shape) { return b.direct(std::move(name), std::move(shape), QType::BF16, residency); };
    VisionWeights out;
    out.patch_embedding      = use("vision/patch_embedding", {h, config.patch_width()}, "vision/patch_input");
    out.patch_embedding_bias = bf16("vision/patch_embedding_bias", {h});
    out.position_embedding   = bf16("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {bf16(p + "norm1_weight", {h}), bf16(p + "norm1_bias", {h})};
        layer.norm2       = {bf16(p + "norm2_weight", {h}), bf16(p + "norm2_bias", {h})};
        layer.query       = use(p + "attention/query", {h, h}, p + "attention_input");
        layer.key         = use(p + "attention/key", {h, h}, p + "attention_input");
        layer.value       = use(p + "attention/value", {h, h}, p + "attention_input");
        layer.query_bias  = bf16(p + "attention/query_bias", {h});
        layer.key_bias    = bf16(p + "attention/key_bias", {h});
        layer.value_bias  = bf16(p + "attention/value_bias", {h});
        layer.output      = use(p + "attention/output", {h, h}, p + "attention_output");
        layer.output_bias = bf16(p + "attention/output_bias", {h});
        layer.fc1         = use(p + "mlp/fc1", {intermediate, h}, p + "mlp_input");
        layer.fc1_bias    = bf16(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = use(p + "mlp/fc2", {h, intermediate}, p + "mlp_activation");
        layer.fc2_bias    = bf16(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const std::uint64_t merger = config.merger_width();
    out.merger_norm     = {bf16("vision/merger/norm_weight", {h}), bf16("vision/merger/norm_bias", {h})};
    out.merger_fc1      = use("vision/merger/fc1", {merger, merger}, "vision/merger/input");
    out.merger_fc1_bias = bf16("vision/merger/fc1_bias", {merger});
    out.merger_fc2      = use("vision/merger/fc2", {target.hidden_size, merger}, "vision/merger/activation");
    out.merger_fc2_bias = bf16("vision/merger/fc2_bias", {target.hidden_size});
    return out;
}

// The pinned groups the encode window streams (vision offload): prelude, each layer, the merger.
qwen3_5::VisionOverlayLayout vision_overlay_layout(const VisionWeights& vision, const std::vector<PendingWeight>& pending,
                                                   const artifact::MaterializationPlan& plan) {
    const auto handles = [&](std::initializer_list<WeightId> ids) {
        std::vector<artifact::ObjectHandle> out;
        for (const WeightId id : ids) {
            for (const auto& part : pending.at(id.index).reference.binding.parts) { out.push_back(part.object); }
        }
        return out;
    };
    std::vector<std::vector<artifact::ObjectHandle>> layers;
    for (const VisionBlockWeights& s : vision.layers) {
        layers.push_back(handles({s.norm1.weight, s.norm1.bias, s.norm2.weight, s.norm2.bias, s.query, s.key, s.value,
                                  s.query_bias, s.key_bias, s.value_bias, s.output, s.output_bias, s.fc1, s.fc1_bias,
                                  s.fc2, s.fc2_bias}));
    }
    return qwen3_5::make_vision_overlay_layout(
        plan, handles({vision.patch_embedding, vision.patch_embedding_bias, vision.position_embedding}), layers,
        handles({vision.merger_norm.weight, vision.merger_norm.bias, vision.merger_fc1, vision.merger_fc1_bias,
                 vision.merger_fc2, vision.merger_fc2_bias}));
}

FrontendResources bind_resources(artifact::Binder& binder, const Config& config) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    if (config.vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
    }
    // The shared frontend validates the token domain against the embedding rows and, with vision,
    // the preprocessors against the tower.
    qwen3_5::Config domain;
    domain.text.vocab_size = config.text.vocab_size;
    domain.vision          = config.vision;
    qwen3_5::parse_resources(out, domain);
    return out;
}

// alpha = fl32(weight_scale_2 * input_scale) per matrix, from the stored words (design §6.1).
std::vector<ops::offloaded_moe::ExpertScales> expert_scales(const ExpertBankPlanes& planes,
                                                           const artifact::HostValues& inputs,
                                                           const std::string& name) {
    if (inputs.format != QType::FP32 || inputs.elements != std::uint64_t(planes.experts) * 3) {
        throw ArtifactError(name + ": expert input scales must be FP32 [experts, 3]");
    }
    std::vector<ops::offloaded_moe::ExpertScales> out(planes.experts);
    const auto* g = reinterpret_cast<const float*>(inputs.data.data());
    for (std::uint32_t e = 0; e < planes.experts; ++e) {
        const float* m = planes.multipliers + 3 * std::size_t(e);
        const float* s = g + 3 * std::size_t(e);
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(m[i]) || m[i] <= 0 || !std::isfinite(s[i]) || s[i] <= 0) {
                throw ArtifactError(name + ": expert scales must be positive finite FP32");
            }
        }
        // One IEEE binary32 product each (this target compiles without FP contraction).
        out[e] = {s[0], s[1], s[2], m[0] * s[0], m[1] * s[1], m[2] * s[2]};
    }
    return out;
}

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    TextWeights weights;
    std::vector<PendingWeight> pending;
    std::vector<artifact::HostValues> input_scales; // [layer]
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::optional<qwen3_5::VisionOverlayLayout> vision_overlay; // vision offload
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

std::uint64_t LoadPlan::pinned_other_bytes() const {
    // Besides the expert banks: the token embedding (bind_text) and, with vision offload, the tower.
    const auto& c = impl_->config.text;
    std::uint64_t bytes = std::uint64_t(c.vocab_size) * c.hidden_size * 2U;
    if (impl_->vision_overlay) {
        const auto& v = *impl_->vision_overlay;
        bytes += v.prelude.bytes + v.merger.bytes;
        for (const auto& layer : v.layers) { bytes += layer.bytes; }
    }
    return bytes;
}

std::uint64_t LoadPlan::device_bytes() const { return impl_->materialization.device_capacity_bytes; }

std::uint32_t LoadPlan::public_token_count() const {
    return static_cast<std::uint32_t>(impl_->resources.public_token_count);
}

std::uint64_t LoadPlan::pinned_expert_bytes() const {
    const std::uint64_t pinned = impl_->materialization.pinned_capacity_bytes;
    const std::uint64_t other  = pinned_other_bytes();
    return pinned > other ? pinned - other : 0;
}

void LoadPlan::set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes) {
    impl_->materialization.host_reserve_bytes = reserve_bytes;
    impl_->materialization.later_pinned_bytes = later_pinned_bytes;
}

bool is_qwen4_exp(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    if (!config.contains("architectures") || !config.contains("model_type")) { return false; }
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1 || !names[0].is_string() ||
        !config.at("model_type").is_string()) {
        return false;
    }
    try {
        return resolve_architecture(names[0].get<std::string>(),
                                    config.at("model_type").get<std::string>()) ==
               Architecture::Qwen4Exp;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config);
    Bindings bindings(binder);
    out->weights = bind_text(bindings, out->config.text,
                             options.stream_experts ? Residency::Streamed : Residency::HostPinned);
    if (out->config.vision) {
        out->weights.vision = bind_vision(bindings, *out->config.vision, out->config.text,
                                          options.overlay_vision() ? Residency::HostPinned : Residency::Device);
    }
    if (out->config.mtp) {
        out->weights.mtp = bind_mtp(bindings, out->config.text);
        if (out->config.proposal_rows != 0) {
            const std::uint64_t rows = out->config.proposal_rows;
            ProposalWeights proposal;
            proposal.head      = bindings.parameter("proposal/head", {rows, out->config.text.hidden_size},
                                                    "mtp/final_hidden");
            proposal.token_ids = bindings.direct("proposal/token_ids", {rows}, QType::INT32);
            const auto ids = binder.values(bindings.weights.at(proposal.token_ids.index).reference.binding,
                                           QType::INT32)
                                 .integers();
            std::vector<bool> seen(out->config.text.vocab_size, false);
            for (const auto id : ids) {
                if (id < 0 || std::uint32_t(id) >= out->config.text.vocab_size || seen[std::size_t(id)]) {
                    throw ArtifactError("proposal token ids must be unique vocabulary ids");
                }
                seen[std::size_t(id)] = true;
            }
            out->weights.proposal = proposal;
        }
    }
    for (const auto& layer : out->weights.layers) {
        out->input_scales.push_back(
            binder.values(bindings.weights.at(layer.moe.input_scales.index).reference.binding,
                          QType::FP32));
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    if (out->weights.vision && options.overlay_vision()) {
        out->vision_overlay = vision_overlay_layout(*out->weights.vision, out->pending, out->materialization);
    }
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(Architecture::Qwen4Exp)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
             std::vector<ExpertBank> banks, FrontendResources resources, InstanceInfo info,
             std::optional<qwen3_5::VisionOverlayAssets> overlay_vision, artifact::MaterializedArtifact backing,
             std::optional<ExpertStore> expert_store)
    : backing_(std::move(backing)), expert_store_(std::move(expert_store)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      resources_(std::move(resources)), info_(std::move(info)), overlay_vision_(std::move(overlay_vision)) {}

Model::~Model() = default;

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    const artifact::Reader& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader, std::move(data->materialization), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(data->pending.size());
    for (auto& item : data->pending) {
        BoundWeight weight;
        weight.name   = item.reference.name;
        weight.policy = item.policy;
        if (item.reference.residency != Residency::Values) {
            weight.view = artifact::bind_view(item.reference, backing);
        } else {
            weight.view.shape = item.reference.shape;
        }
        bound.push_back(std::move(weight));
    }
    std::vector<ExpertBank> banks;
    const auto& layers = data->weights.layers;
    // Streamed banks (the SSD tier): the records stay in the artifact; each layer's multipliers
    // (the scale tail) are read here into Model memory.
    const bool streamed = data->options.stream_experts;
    std::vector<artifact::ObjectHandle> bank_objects;
    std::vector<std::vector<float>> multipliers;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& view = bound.at(layers[i].moe.experts.index).view;
        if (view.parts.size() != 1 || !is_complete_weight(view)) {
            throw ArtifactError(bound.at(layers[i].moe.experts.index).name +
                                ": an expert bank must be one complete parent");
        }
        ExpertBank bank;
        if (streamed) {
            const auto& reference = data->pending.at(layers[i].moe.experts.index).reference;
            const auto object     = reference.binding.parts.front().object;
            const auto& parent    = *view.parts.front().parent;
            std::vector<float> words(parent.geometry.scale_bytes / sizeof(float));
            reader.read_into(artifact::object_offset(reader.directory().object(object)) + parent.geometry.scale_offset,
                             std::as_writable_bytes(std::span(words)));
            bank.planes = expert_bank_layout(parent.geometry, words.data());
            bank_objects.push_back(object);
            multipliers.push_back(std::move(words));
        } else {
            bank.planes = expert_bank_planes(*view.parts.front().parent);
        }
        if (bank.planes.experts != data->config.text.moe.experts ||
            bank.planes.hidden != data->config.text.hidden_size ||
            bank.planes.intermediate != data->config.text.moe.intermediate) {
            throw ArtifactError("expert bank geometry differs from the text config");
        }
        bank.scales = expert_scales(bank.planes, data->input_scales.at(i),
                                    bound.at(layers[i].moe.input_scales.index).name);
        banks.push_back(std::move(bank));
    }
    std::optional<ExpertStore> store;
    if (streamed) {
        const auto& c = data->config.text;
        store.emplace(backing.stream_source(), bank_objects, banks.front().planes.record_stride, c.moe.experts,
                      std::move(multipliers)); // the vectors move, so the planes' pointers stay valid
    }
    std::optional<qwen3_5::VisionOverlayAssets> overlay;
    if (data->vision_overlay) {
        const auto pinned = backing.pinned_bytes_range();
        if (pinned.empty()) { throw ArtifactError("vision offload requires a pinned materialization"); }
        overlay = qwen3_5::VisionOverlayAssets{.pool         = nullptr,
                                               .pinned_block = pinned.data(),
                                               .pinned_bytes = pinned.size(),
                                               .ladder_bytes = 0,
                                               .layout       = std::move(*data->vision_overlay)};
    }
    return std::unique_ptr<Model>(new Model(std::move(data->config), data->options,
                                            std::move(data->weights), std::move(bound),
                                            std::move(banks), std::move(data->resources),
                                            std::move(data->info), std::move(overlay), std::move(backing),
                                            std::move(store)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen4_exp
