#include "models/inkling2/load.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/weight_view.h"

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::inkling2 {
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

// Logical demands of the model; each parameter is declared exactly once, device-resident.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {}, std::optional<QType> format = {}) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), Residency::Device, format);
        if (!input.empty()) {
            const auto& use = binder_.use(name, input);
            if (!use.activation_policy) {
                throw ArtifactError(name + "@" + std::string(input) + ": missing activation policy");
            }
            if (!use.auxiliaries.empty()) { throw ArtifactError(name + ": Inkling 2 weights carry no Use auxiliaries"); }
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

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, const std::string& p, bool full) {
    const std::uint64_t h = c.hidden_size, q = c.query_width(), k = c.key_width(), r = c.relative_width();
    const std::string a = p + "attention/", xq = a + "query_input", xk = a + "key_input", xv = a + "value_input";
    AttentionWeights out;
    out.query    = b.parameter(a + "query", {q, h}, xq);
    out.relative = b.parameter(a + "relative", {r, h}, xq);
    out.gate     = b.parameter(a + "gate", {q, h}, xq);
    out.key      = b.parameter(a + "key", {k, h}, xk);
    out.value    = b.parameter(a + "value", {k, h}, xv);
    out.sink     = b.parameter(a + "sink", {c.heads, h}, xq);
    if (full) {
        out.key_taps   = b.parameter(a + "key_taps", {std::uint64_t{c.kv_heads} * c.conv_kernel, h}, xk);
        out.value_taps = b.parameter(a + "value_taps", {std::uint64_t{c.kv_heads} * c.conv_kernel, h}, xv);
    }
    out.output         = b.parameter(a + "output", {h, q}, a + "gated");
    out.query_scale    = b.direct(a + "query_scale", {c.heads, c.head_dim});
    out.relative_scale = b.direct(a + "relative_scale", {c.heads, c.relative_dim});
    out.relative_table = b.direct(a + "relative_table", {c.relative_dim, c.relative_extent});
    out.key_conv       = b.direct(a + "key_conv", {c.conv_kernel, k});
    out.value_conv     = b.direct(a + "value_conv", {c.conv_kernel, k});
    return out;
}

MessageWeights bind_messages(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h = c.hidden_size, rank = c.loop.message_rank;
    constexpr std::array<const char*, 4> kinds = {"query", "key", "value", "mlp"};
    MessageWeights out;
    out.queries = b.direct(p + "message_queries", {4, h}, QType::FP32);
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const std::string g = p + "messages/" + kinds[i] + "/";
        out.gates[i].down   = b.parameter(g + "down", {rank, 2 * h}, g + "pair");
        out.gates[i].forget = b.parameter(g + "forget", {h, rank}, g + "latent");
        out.gates[i].gain   = b.parameter(g + "gain", {h, rank}, g + "latent");
    }
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& p, ExpertFormat format) {
    const std::uint64_t h = c.hidden_size, e = c.experts;
    const std::string m = p + "moe/", input = m + "input";
    MoeWeights out;
    out.router      = b.parameter(m + "router", {e, h}, input);
    out.router_bias = b.direct(m + "router_bias", {e}, QType::FP32);
    const QType bank = format == ExpertFormat::Nvfp4FourOverSix ? QType::NVFP4_MUL : QType::FP8_E4M3FN_G32_E8M0;
    out.experts      = b.parameter(m + "experts", {e, h, c.intermediate}, input, bank);
    if (format == ExpertFormat::Nvfp4FourOverSix) {
        out.expert_amax         = b.direct(m + "expert_amax", {e, 2}, QType::FP32);
        out.expert_denominators = b.direct(m + "expert_denominators", {2}, QType::FP32);
    }
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    TextWeights out;
    out.token_embedding = b.direct("text/token_embedding", {v, h});
    out.embedding_norm  = b.direct("text/embedding_norm", {h});
    out.output_head     = b.parameter("text/output_head", {v, h}, "text/final_hidden");
    out.final_norm      = b.direct("text/final_norm", {h});
    out.output_query    = b.direct("text/output_query", {h}, QType::FP32);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const bool full     = c.layer_types[i] == AttentionKind::Full;
        LayerWeights layer;
        if (i > 0) {
            layer.attention_mix_query = b.direct(p + "attention_mix_query", full ? Shape{3, h} : Shape{h}, QType::FP32);
        }
        layer.mlp_mix_query = b.direct(p + "mlp_mix_query", {h}, QType::FP32);
        if (full) {
            layer.query_norm = b.direct(p + "query_norm", {h});
            layer.key_norm   = b.direct(p + "key_norm", {h});
            layer.value_norm = b.direct(p + "value_norm", {h});
        } else {
            layer.attention_norm = b.direct(p + "attention_norm", {h});
        }
        layer.mlp_norm       = b.direct(p + "mlp_norm", {h});
        layer.attention      = bind_attention(b, c, p, full);
        layer.attention_conv = b.direct(p + "attention_conv", {c.conv_kernel, h});
        layer.mlp_conv       = b.direct(p + "mlp_conv", {c.conv_kernel, h});
        if (c.in_core(i)) { layer.messages = bind_messages(b, c, p); }
        layer.moe = bind_moe(b, c, p, c.expert_formats[i]);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

// The stored experts must carry the arithmetic the config names: their Use permits it, and an A16Only Use (a
// recipe that withheld the trained activation rule) is refused rather than run in another arithmetic.
void check_expert_uses(const std::vector<PendingWeight>& pending, const TextWeights& weights, const TextConfig& c) {
    for (std::size_t i = 0; i < weights.layers.size(); ++i) {
        const auto& bank     = pending.at(weights.layers[i].moe.experts.index);
        const bool nvfp4     = c.expert_formats[i] == ExpertFormat::Nvfp4FourOverSix;
        const auto permitted = nvfp4 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::AllowA8;
        if (bank.policy != permitted) {
            throw ArtifactError(bank.reference.name + ": the trained " + (nvfp4 ? "W4A4" : "W8A8") +
                                " expert arithmetic needs its Use to permit " + (nvfp4 ? "AllowA4" : "AllowA8"));
        }
    }
}

// The artifact objects load-time packing takes (LoadOptions::pack_bf16; models/bf16_packing.h): the BF16 matrices
// whose packed route measured faster in the decode round. Every projection's reader takes bf16_p12 (the fused
// projections and ops::linear), but packed alone at 4K, C = 1 (11-21, the round 6.305 ms unpacked) the head saves
// 0.129 ms, while each visit's K2 parent costs 0.070 ms more, W_o 0.044 ms and the message gates 0.269 ms, although
// the op sweep has K2 and W_o faster at one column. So only the head packs (none under q8_weights, whose Q8 head
// measured faster still).
//
// The artifact objects whose every parameter is selected, device-resident, and a BF16 [n, k] matrix with k a multiple
// of `k_multiple`.
std::vector<artifact::ObjectHandle> objects_of(const std::vector<bool>& selected, const std::vector<PendingWeight>& pending,
                                               const artifact::Reader& reader, std::uint64_t k_multiple) {
    std::map<std::size_t, bool> objects; // object -> every parameter on it selected
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const bool ok = selected[i] && pending[i].reference.residency == Residency::Device;
        for (const auto& part : pending[i].reference.binding.parts) {
            auto [it, fresh] = objects.emplace(part.object.index, ok);
            if (!fresh) { it->second = it->second && ok; }
        }
    }
    std::vector<artifact::ObjectHandle> out;
    for (const auto& [index, ok] : objects) {
        const artifact::ObjectHandle handle{index};
        const auto& g = reader.geometry(handle);
        if (ok && g.format == QType::BF16 && g.layout == QuantLayout::Contiguous && g.shape.size() == 2 &&
            g.shape[1] % k_multiple == 0 &&
            g.shape[0] <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            out.push_back(handle);
        }
    }
    return out;
}

std::vector<artifact::ObjectHandle> packable_objects(const TextWeights& w, const std::vector<PendingWeight>& pending,
                                                     const artifact::Reader& reader, const LoadOptions& options) {
    std::vector<bool> packed(pending.size(), false);
    if (w.output_head.valid() && !options.q8_weights) { packed.at(w.output_head.index) = true; }
    return objects_of(packed, pending, reader, 8);
}

// The artifact objects --q8-weights converts (LoadOptions::q8_weights; models/q8_weights.h), the groups measured
// faster as 8-bit weights in the decode round (2026-11-21, 4K context, captured round, against the shipped route):
// each attention's fused input projections (the K2 parent and the sink/taps parent beside it, one launch) -7.6 % at
// one column, -5.6 % at two, -1.3 % at eight; the head (in place of its bf16_p12 packing) -1.6 %, -1.4 %, -0.2 %.
// W_o (-2.5 %, -1.4 %, +2.7 %; +6.7 % of its prefill time) and the message gates' downs (-0.1 %, +0.1 %, +0.1 %) stay
// BF16.
std::vector<artifact::ObjectHandle> q8_objects(const TextWeights& w, const std::vector<PendingWeight>& pending,
                                               const artifact::Reader& reader) {
    std::vector<bool> selected(pending.size(), false);
    const auto select = [&](WeightId id) {
        if (id.valid()) { selected.at(id.index) = true; }
    };
    for (const auto& layer : w.layers) {
        const auto& a = layer.attention;
        for (const WeightId id : {a.query, a.relative, a.gate, a.key, a.value, a.sink, a.key_taps, a.value_taps}) {
            select(id);
        }
    }
    select(w.output_head);
    return objects_of(selected, pending, reader, 32);
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
    out.tokenizer              = std::make_shared<const text::Tokenizer>(
        text::TokenizerResources{.tokenizer_json         = out.tokenizer_json,
                                 .tokenizer_config_json  = out.tokenizer_config_json,
                                 .generation_config_json = out.generation_config_json,
                                 .pipeline               = text::TokenizerPipeline::Inkling});
    const auto count = out.tokenizer->vocab_size();
    if (count != config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw ArtifactError("the tokenizer's public domain must be exactly the model's vocabulary");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : out.tokenizer->default_stop_token_ids()) {
        if (!out.tokenizer->is_valid_token(token)) { throw ArtifactError("stop token is outside the vocabulary"); }
    }
    return out;
}

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    TextWeights weights;
    std::vector<PendingWeight> pending;
    std::vector<artifact::ObjectHandle> packed_objects; // LoadOptions::pack_bf16: loaded transient, packed at load
    std::uint64_t packed_bytes = 0;
    std::vector<artifact::ObjectHandle> q8_objects; // LoadOptions::q8_weights: loaded transient, converted at load
    std::uint64_t q8_bytes = 0;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }
const FrontendResources& LoadPlan::resources() const { return impl_->resources; }
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->materialization; }
std::uint64_t LoadPlan::device_bytes() const {
    return impl_->materialization.device_capacity_bytes + impl_->packed_bytes + impl_->q8_bytes;
}
std::uint32_t LoadPlan::public_token_count() const { return impl_->resources.public_token_count; }

bool is_inkling2(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    return config.contains("architectures") && config.at("architectures") == artifact::Json::array({"Inkling2LoopForCausalLM"});
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config.text);
    Bindings bindings(binder);
    out->weights = bind_text(bindings, out->config.text);
    check_expert_uses(bindings.weights, out->weights, out->config.text);
    out->pending = std::move(bindings.weights);
    if (options.pack_bf16) {
        out->packed_objects = packable_objects(out->weights, out->pending, reader, options);
        for (const auto object : out->packed_objects) {
            binder.mark_device_transient(object);
            out->packed_bytes += models::packed_estimate(reader.geometry(object));
        }
    }
    if (options.q8_weights) {
        out->q8_objects = q8_objects(out->weights, out->pending, reader);
        for (const auto object : out->q8_objects) {
            binder.mark_device_transient(object);
            out->q8_bytes += models::q8_bytes(reader.geometry(object));
        }
    }
    out->materialization = std::move(binder).finish();
    out->info.name            = reader.directory().metadata.value("name", std::string("Inkling2LoopForCausalLM"));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing, PackedWeights packed,
             Q8Weights q8)
    : backing_(std::move(backing)), packed_(std::move(packed)), q8_(std::move(q8)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device, const StartupObserver* observer) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data                      = std::move(plan.impl_);
    const artifact::Reader& reader = *data->materialization.source;
    auto backing                   = artifact::materialize(reader, std::move(data->materialization), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(data->pending.size());
    for (auto& item : data->pending) {
        BoundWeight weight;
        weight.name   = item.reference.name;
        weight.policy = item.policy;
        weight.view   = artifact::bind_view(item.reference, backing);
        bound.push_back(std::move(weight));
    }
    PackedWeights packed = models::pack_bf16_objects(backing, data->packed_objects, {}, device);
    Q8Weights q8         = models::quantize_q8_objects(backing, data->q8_objects, device);
    if (!data->packed_objects.empty() || !data->q8_objects.empty()) {
        for (auto& weight : bound) {
            models::repoint(weight.view, packed);
            models::repoint(weight.view, q8);
        }
        backing.release_transient();
    }
    return std::unique_ptr<Model>(new Model(std::move(data->config), data->options, std::move(data->weights),
                                            std::move(bound), std::move(data->resources), std::move(data->info),
                                            std::move(backing), std::move(packed), std::move(q8)));
}

} // namespace infernix::models::inkling2
