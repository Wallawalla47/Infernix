#include "models/qwen4_exp/load.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
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

    WeightId direct(std::string name, Shape shape, QType format = QType::BF16) {
        return parameter(std::move(name), std::move(shape), {}, format);
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

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, e = c.moe.experts, s = c.moe.shared_intermediate;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    MoeWeights out;
    out.router       = b.parameter(p + "router", {e, h}, input);
    out.shared_score = b.parameter(p + "shared_score", {1, h}, input);
    out.shared_gate  = b.parameter(p + "shared/gate", {s, h}, input);
    out.shared_up    = b.parameter(p + "shared/up", {s, h}, input);
    out.shared_down  = b.parameter(p + "shared/down", {h, s}, p + "shared/product");
    out.experts      = b.parameter(p + "experts", {e, h, c.moe.intermediate}, input, QType::NVFP4_MUL,
                                   Residency::HostPinned);
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

TextWeights bind_text(Bindings& b, const TextConfig& c) {
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
        block.moe    = bind_moe(b, c, prefix);
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

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    // The shared frontend validates the token domain against the embedding rows only.
    qwen3_5::Config domain;
    domain.text.vocab_size = config.vocab_size;
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
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
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
    out->resources = bind_resources(binder, out->config.text);
    Bindings bindings(binder);
    out->weights = bind_text(bindings, out->config.text);
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
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(Architecture::Qwen4Exp)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
             std::vector<ExpertBank> banks, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
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
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& view = bound.at(layers[i].moe.experts.index).view;
        if (view.parts.size() != 1 || !is_complete_weight(view)) {
            throw ArtifactError(bound.at(layers[i].moe.experts.index).name +
                                ": an expert bank must be one complete parent");
        }
        ExpertBank bank;
        bank.planes = expert_bank_planes(*view.parts.front().parent);
        if (bank.planes.experts != data->config.text.moe.experts ||
            bank.planes.hidden != data->config.text.hidden_size ||
            bank.planes.intermediate != data->config.text.moe.intermediate) {
            throw ArtifactError("expert bank geometry differs from the text config");
        }
        bank.scales = expert_scales(bank.planes, data->input_scales.at(i),
                                    bound.at(layers[i].moe.input_scales.index).name);
        banks.push_back(std::move(bank));
    }
    return std::unique_ptr<Model>(new Model(std::move(data->config), data->options,
                                            std::move(data->weights), std::move(bound),
                                            std::move(banks), std::move(data->resources),
                                            std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen4_exp
