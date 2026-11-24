#include "models/inkling2/load.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/weight_view.h"

#include <algorithm>
#include <initializer_list>
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

// Logical demands of the model; each parameter is declared exactly once, device-resident (or at `residency` while it
// is set: an offloaded vision tower's HostPinned).
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {}, std::optional<QType> format = {}) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, format);
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
    Residency residency = Residency::Device;

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

// `embedding_host` (--embedding-host): the token embedding (0.86 GiB) in pinned host memory, its rows gathered
// zero-copy by ik::embed (one row a decode token; a prefill chunk's rows over the link), as Qwen3.5's option.
TextWeights bind_text(Bindings& b, const TextConfig& c, bool embedding_host) {
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    TextWeights out;
    b.residency         = embedding_host ? Residency::HostPinned : Residency::Device;
    out.token_embedding = b.direct("text/token_embedding", {v, h});
    b.residency         = Residency::Device;
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

// The DSpark drafter's parameters (tools/convert/inkling2_loop.py dspark(): names, shapes and the Uses' inputs).
DSparkWeights bind_dspark(Bindings& b, const DSparkConfig& d, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, v = c.vocab_size, rank = d.markov_rank;
    const std::uint64_t q = std::uint64_t{d.heads} * d.head_dim, kv = std::uint64_t{d.kv_heads} * d.head_dim;
    const std::uint64_t features = d.target_layer_ids.size();
    DSparkWeights out;
    out.feature_weights = b.direct("dspark/feature_weights", {d.layers + 1, features});
    out.mask_embedding  = b.direct("dspark/mask_embedding", {h});
    out.final_norm      = b.direct("dspark/final_norm", {h});
    for (std::uint32_t i = 0; i < d.layers; ++i) {
        const std::string p = "dspark/layers/" + std::to_string(i) + "/", a = p + "attention/", f = p + "mlp/";
        DSparkLayerWeights l;
        l.context_norm        = b.direct(p + "context_norm", {h});
        l.input_norm          = b.direct(p + "input_norm", {h});
        l.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        l.query               = b.parameter(a + "query", {q, h}, p + "hidden_input");
        l.key                 = b.parameter(a + "key", {kv, h}, p + "hidden_input");
        l.value               = b.parameter(a + "value", {kv, h}, p + "hidden_input");
        l.context_key         = b.parameter(a + "context_key", {kv, h}, p + "context_input");
        l.context_value       = b.parameter(a + "context_value", {kv, h}, p + "context_input");
        l.query_norm          = b.direct(a + "query_norm", {d.head_dim});
        l.key_norm            = b.direct(a + "key_norm", {d.head_dim});
        l.context_key_norm    = b.direct(a + "context_key_norm", {d.head_dim});
        l.output              = b.parameter(a + "output", {h, q}, a + "mixed");
        l.gate                = b.parameter(f + "gate", {d.intermediate, h}, f + "input");
        l.up                  = b.parameter(f + "up", {d.intermediate, h}, f + "input");
        l.down                = b.parameter(f + "down", {h, d.intermediate}, f + "product");
        out.layers.push_back(l);
    }
    const std::string mk = "dspark/markov/";
    out.markov_context_norm    = b.direct(mk + "context_norm", {h});
    out.markov_embedding       = b.direct(mk + "embedding", {v, rank});
    out.markov_projection      = b.parameter(mk + "projection", {3 * rank, 2 * rank + 2 * h}, mk + "input");
    out.markov_projection_bias = b.direct(mk + "projection_bias", {3 * rank});
    out.fold                   = b.parameter(mk + "fold", {v, rank}, mk + "latent");
    out.fold_bias              = b.direct(mk + "fold_bias", {v});
    return out;
}

// The vision tower's parameters (tools/convert/inkling2_loop.py vision(): names, shapes and the Uses' inputs).
VisionWeights bind_vision(Bindings& b, const VisionConfig& c) {
    const std::uint64_t h = c.hidden_size, inter = c.intermediate, grid = c.position_grid;
    VisionWeights out;
    out.patch_embedding    = b.parameter("vision/patch_embedding", {h, c.patch_width()}, "vision/patch_input");
    out.position_embedding = b.direct("vision/position_embedding", {grid * grid, h});
    out.class_row          = b.direct("vision/class_row", {h});
    out.pre_norm           = b.direct("vision/pre_norm", {h});
    out.pre_norm_bias      = b.direct("vision/pre_norm_bias", {h});
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "vision/layers/" + std::to_string(i) + "/", a = p + "attention/", m = p + "mlp/";
        VisionLayerWeights l;
        l.attention_norm      = b.direct(p + "attention_norm", {h});
        l.attention_norm_bias = b.direct(p + "attention_norm_bias", {h});
        l.query               = b.parameter(a + "query", {h, h}, a + "input");
        l.key                 = b.parameter(a + "key", {h, h}, a + "input");
        l.value               = b.parameter(a + "value", {h, h}, a + "input");
        l.query_bias          = b.direct(a + "query_bias", {h});
        l.key_bias            = b.direct(a + "key_bias", {h});
        l.value_bias          = b.direct(a + "value_bias", {h});
        l.output              = b.parameter(a + "output", {h, h}, a + "attended");
        l.output_bias         = b.direct(a + "output_bias", {h});
        l.attention_scale     = b.direct(p + "attention_scale", {h});
        l.mlp_norm            = b.direct(p + "mlp_norm", {h});
        l.mlp_norm_bias       = b.direct(p + "mlp_norm_bias", {h});
        l.up                  = b.parameter(m + "up", {inter, h}, m + "input");
        l.up_bias             = b.direct(m + "up_bias", {inter});
        l.down                = b.parameter(m + "down", {h, inter}, m + "hidden");
        l.down_bias           = b.direct(m + "down_bias", {h});
        l.mlp_scale           = b.direct(p + "mlp_scale", {h});
        out.layers.push_back(l);
    }
    out.projector_in  = b.parameter("vision/projector/in", {c.projection_hidden, c.merged_width()},
                                    "vision/projector/merged");
    out.projector_out = b.parameter("vision/projector/out", {c.output_size, c.projection_hidden},
                                    "vision/projector/hidden");
    return out;
}

// --vision-offload on: the tower's groups in the pinned block (design §15.5). A group's distinct objects in the order
// listed, each at the next 16-byte boundary of its device region (a parent shared by several weights, as q/k/v, is
// staged once).
constexpr std::size_t kStagedObjectAlignment = 16;

VisionStagingGroup staging_group(const std::vector<PendingWeight>& pending,
                                 const std::map<std::size_t, artifact::PinnedPlacement>& pinned,
                                 std::initializer_list<WeightId> ids) {
    VisionStagingGroup group;
    std::vector<std::size_t> seen;
    for (const WeightId id : ids) {
        for (const auto& part : pending.at(id.index).reference.binding.parts) {
            if (std::find(seen.begin(), seen.end(), part.object.index) != seen.end()) { continue; }
            seen.push_back(part.object.index);
            const auto it = pinned.find(part.object.index);
            if (it == pinned.end()) { throw ArtifactError(pending.at(id.index).reference.name + ": not pinned"); }
            group.bytes = (group.bytes + kStagedObjectAlignment - 1) / kStagedObjectAlignment * kStagedObjectAlignment;
            group.segments.push_back({static_cast<std::size_t>(it->second.offset), group.bytes,
                                      static_cast<std::size_t>(it->second.bytes)});
            group.bytes += static_cast<std::size_t>(it->second.bytes);
        }
    }
    return group;
}

VisionHostWeights vision_host_layout(const VisionWeights& w, const std::vector<PendingWeight>& pending,
                                     const artifact::MaterializationPlan& plan) {
    std::map<std::size_t, artifact::PinnedPlacement> pinned;
    for (const artifact::PinnedPlacement& placement : plan.pinned_objects) { pinned.emplace(placement.object.index, placement); }
    VisionHostWeights out;
    out.resident = staging_group(pending, pinned,
                                 {w.patch_embedding, w.position_embedding, w.class_row, w.pre_norm, w.pre_norm_bias,
                                  w.projector_in, w.projector_out});
    for (const VisionLayerWeights& l : w.layers) {
        out.layers.push_back(staging_group(
            pending, pinned,
            {l.attention_norm, l.attention_norm_bias, l.query, l.key, l.value, l.query_bias, l.key_bias, l.value_bias,
             l.output, l.output_bias, l.attention_scale, l.mlp_norm, l.mlp_norm_bias, l.up, l.up_bias, l.down,
             l.down_bias, l.mlp_scale}));
        out.slot_bytes = std::max(out.slot_bytes, out.layers.back().bytes);
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
// W_o since 2026-11-22 (re-timed on the fused projections' kernel at context 128, three runs each: 4.677 -> 4.571 ms
// at one column, 5.878 -> 5.796 at two, 9.428 -> 9.256 at eight; on the perplexity corpus's quick set, against one
// earlier BF16 reference, KL 1.064e-2 with it and 1.066e-2 without, top-1 96.01 % and 96.05 %; against the current
// build's BF16 the whole set reads KL 1.05e-2, top-1 96.06 %, PPL 4.03419 vs 4.03144). The message gates' downs
// (-0.1 %, +0.1 %, +0.1 %) stay BF16.
std::vector<artifact::ObjectHandle> q8_objects(const TextWeights& w, const std::vector<PendingWeight>& pending,
                                               const artifact::Reader& reader) {
    std::vector<bool> selected(pending.size(), false);
    const auto select = [&](WeightId id) {
        if (id.valid()) { selected.at(id.index) = true; }
    };
    for (const auto& layer : w.layers) {
        const auto& a = layer.attention;
        for (const WeightId id :
             {a.query, a.relative, a.gate, a.key, a.value, a.sink, a.key_taps, a.value_taps, a.output}) {
            select(id);
        }
    }
    select(w.output_head);
    return objects_of(selected, pending, reader, 32);
}

// The drafter's matrices converted to 8-bit weights whenever DSpark is bound (not an opt-in: only the drafts depend on
// them, and a draft is kept only when it equals the target's own token, so no route's output does; STRATEGIES-REVIEW
// D1, design §13.6): each layer's dense projections (q/k/v, the context's k/v, o, gate/up/down: the round's 0.52 GB
// halved), and the Markov head's Fold [V, 256], 102 -> 54 MB, so it stays in L2 across a round's seven steps.
std::vector<artifact::ObjectHandle> drafter_q8_objects(const DSparkWeights& d, const std::vector<PendingWeight>& pending,
                                                       const artifact::Reader& reader) {
    std::vector<bool> selected(pending.size(), false);
    const auto select = [&](WeightId id) {
        if (id.valid()) { selected.at(id.index) = true; }
    };
    for (const DSparkLayerWeights& l : d.layers) {
        for (const WeightId id : {l.query, l.key, l.value, l.context_key, l.context_value, l.output, l.gate, l.up, l.down}) {
            select(id);
        }
    }
    select(d.fold);
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
    std::vector<std::int32_t> draft_ids; // --lm-head-draft: the shortlist's tokens, ascending
    std::uint64_t draft_bytes = 0;
    std::vector<artifact::ObjectHandle> q8_objects; // LoadOptions::q8_weights: loaded transient, converted at load
    std::uint64_t q8_bytes = 0;
    artifact::MaterializationPlan materialization;
    std::optional<VisionHostWeights> vision_host; // --vision-offload on (block filled at materialization)
    FrontendResources resources;
    InstanceInfo info;
    std::optional<LoraBinding> lora;             // LoadOptions::lora
    std::optional<AblateSidecarBinding> ablate; // LoadOptions::ablate
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }
const FrontendResources& LoadPlan::resources() const { return impl_->resources; }
const artifact::MaterializationPlan& LoadPlan::materialization() const { return impl_->materialization; }
const LoraBinding* LoadPlan::lora() const { return impl_->lora ? &*impl_->lora : nullptr; }
const AblateSidecarBinding* LoadPlan::ablate() const { return impl_->ablate ? &*impl_->ablate : nullptr; }
std::uint64_t LoadPlan::device_bytes() const {
    return impl_->materialization.device_capacity_bytes + impl_->packed_bytes + impl_->q8_bytes + impl_->draft_bytes;
}
std::uint32_t LoadPlan::public_token_count() const { return impl_->resources.public_token_count; }
const std::optional<VisionHostWeights>& LoadPlan::vision_host() const { return impl_->vision_host; }

bool is_inkling2(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    return config.contains("architectures") && config.at("architectures") == artifact::Json::array({"Inkling2LoopForCausalLM"});
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    if (!options.lora.empty()) { out->lora = bind_lora(options.lora, out->config.text); }
    if (!options.ablate.empty()) {
        out->ablate = bind_ablate(options.ablate, out->config.text.num_hidden_layers, out->config.text.hidden_size);
    }
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config.text);
    Bindings bindings(binder);
    out->weights = bind_text(bindings, out->config.text, options.embedding_host);
    if (out->config.dspark) { out->weights.dspark = bind_dspark(bindings, *out->config.dspark, out->config.text); }
    if (options.proposal_enabled()) {
        // --lm-head-draft: the drafter's shortlist, the ranking's first rows (their tables are built at load).
        if (!out->config.dspark) { throw ArtifactError("--lm-head-draft needs the artifact's DSpark drafter"); }
        if (!binder.contains("proposal/token_rank")) {
            throw ArtifactError("--lm-head-draft: this artifact has no token ranking (proposal/token_rank)");
        }
        const WeightId rank = bindings.direct("proposal/token_rank", {out->config.text.vocab_size}, QType::INT32);
        const std::vector<std::int32_t> order =
            binder.values(bindings.weights.at(rank.index).reference.binding, QType::INT32).integers();
        out->draft_ids   = draft_head_ids(order);
        out->draft_bytes = draft_head_bytes(static_cast<std::int32_t>(out->draft_ids.size()),
                                            static_cast<std::int32_t>(out->config.text.hidden_size), 256);
    }
    if (out->config.vision) {
        // --vision-offload on: the tower in pinned host memory, staged through the tower's device slots per encode.
        bindings.residency  = options.vision_offload ? Residency::HostPinned : Residency::Device;
        out->weights.vision = bind_vision(bindings, *out->config.vision);
        bindings.residency  = Residency::Device;
    }
    check_expert_uses(bindings.weights, out->weights, out->config.text);
    out->pending = std::move(bindings.weights);
    if (options.pack_bf16) {
        out->packed_objects = packable_objects(out->weights, out->pending, reader, options);
        for (const auto object : out->packed_objects) {
            binder.mark_device_transient(object);
            out->packed_bytes += models::packed_estimate(reader.geometry(object));
        }
    }
    if (options.q8_weights) { out->q8_objects = q8_objects(out->weights, out->pending, reader); }
    if (out->weights.dspark) {
        for (const auto object : drafter_q8_objects(*out->weights.dspark, out->pending, reader)) {
            if (std::find_if(out->q8_objects.begin(), out->q8_objects.end(),
                             [&](const artifact::ObjectHandle h) { return h.index == object.index; }) == out->q8_objects.end()) {
                out->q8_objects.push_back(object);
            }
        }
    }
    if (!out->q8_objects.empty()) {
        for (const auto object : out->q8_objects) {
            binder.mark_device_transient(object);
            out->q8_bytes += models::q8_bytes(reader.geometry(object));
        }
    }
    out->materialization = std::move(binder).finish();
    if (out->config.vision && options.vision_offload) {
        out->vision_host = vision_host_layout(*out->weights.vision, out->pending, out->materialization);
    }
    out->info.name            = reader.directory().metadata.value("name", std::string("Inkling2LoopForCausalLM"));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing, PackedWeights packed,
             Q8Weights q8, std::optional<DraftHead> draft_head, std::optional<VisionHostWeights> vision_host)
    : backing_(std::move(backing)), packed_(std::move(packed)), q8_(std::move(q8)), draft_head_(std::move(draft_head)),
      config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)),
      vision_host_(std::move(vision_host)) {}

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
    std::optional<DraftHead> draft_head;
    if (!data->draft_ids.empty()) {
        // From the stored BF16 tables, which the views still address (before the repoint below releases them).
        const DSparkWeights& d    = *data->weights.dspark;
        const WeightRegion bias   = contiguous_weight_region(bound.at(d.fold_bias.index).view);
        draft_head                = build_draft_head(native_weight(bound.at(data->weights.output_head.index).view),
                                                     native_weight(bound.at(d.fold.index).view),
                                                     reinterpret_cast<const std::uint16_t*>(bias.parent->device + bias.begin * 2),
                                                     data->draft_ids, device);
    }
    if (!data->packed_objects.empty() || !data->q8_objects.empty()) {
        for (auto& weight : bound) {
            models::repoint(weight.view, packed);
            models::repoint(weight.view, q8);
        }
        backing.release_transient();
    }
    if (data->vision_host) {
        const std::span<const std::byte> pinned = backing.pinned_bytes_range();
        if (pinned.empty()) { throw ArtifactError("--vision-offload on: the tower has no pinned block"); }
        data->vision_host->block = pinned.data();
        void* mapped             = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&mapped, const_cast<std::byte*>(pinned.data()), 0));
        data->vision_host->block_device = static_cast<const std::byte*>(mapped);
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(data->config), data->options, std::move(data->weights),
                                                  std::move(bound), std::move(data->resources), std::move(data->info),
                                                  std::move(backing), std::move(packed), std::move(q8), std::move(draft_head),
                                                  std::move(data->vision_host)));
    model->lora_   = std::move(data->lora);
    model->ablate_ = std::move(data->ablate);
    return model;
}

} // namespace infernix::models::inkling2
